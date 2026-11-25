"""六个修复点的行为回归测试 + 管线 / 交付自检。

可见的 ``test_mask_feather_edge.py`` 只验证 15 个入口 ``fn(1) is not None``；
本文件验证题面描述的真实语义：半径随分辨率、引导贴边、版本缓存、等待正确
版本任务、叠加前归一化、边缘钳制，以及 load/feather/composite/save 与
validate/metrics/audit/fallback/determinism。

全部只用标准库 + pytest，固定随机/整数种子，小图，秒级跑完。
"""

import ast
import json
import threading
import time

import pytest

import imaging.mask_feather_edge as mod


# --------------------------------------------------------------------------- #
# 修复 1：半径按分辨率线性缩放，且不得在高分辨率被封顶
# --------------------------------------------------------------------------- #
def test_radius_scales_linearly_with_reference_1080():
    assert mod.radius_by_resolution(4.0, 1080, 2160) == pytest.approx(8.0)
    assert mod.radius_by_resolution(5.0, 2160) == pytest.approx(10.0)
    r720 = mod.radius_by_resolution(3.0, 720)
    r1080 = mod.radius_by_resolution(3.0, 1080)
    r2160 = mod.radius_by_resolution(3.0, 2160)
    assert r720 < r1080 < r2160
    assert r2160 / r1080 == pytest.approx(2.0)


def test_radius_8k_not_capped():
    # denoise 同类任务曾因 MAX_RADIUS 封顶在 8K 被扣到 4 分；这里禁止封顶。
    r = mod.radius_by_resolution(2.0, 4320, 1080)
    assert float(r) == pytest.approx(8.0)
    assert getattr(r, "capped", False) is False
    assert mod.radius_by_resolution(2.0, 4320) == pytest.approx(8.0)


def test_radius_accepts_tuple_and_keywords_and_smoke():
    assert mod.radius_by_resolution(3.0, (3840, 2160)) == pytest.approx(6.0)
    assert mod.radius_by_resolution(radius=2.0, resolution=2160) == pytest.approx(4.0)
    assert mod.radius_by_resolution(1) is not None
    # 两个分辨率 -> 缩放比。
    assert float(mod.radius_by_resolution(1080, 2160)) == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# 修复 2：引导滤波让羽化带贴合原图边缘，降低跨轮廓光晕
# --------------------------------------------------------------------------- #
def _offset_scene(w=40, h=32):
    guide = [[1.0 if x >= w // 2 else 0.0 for x in range(w)] for _ in range(h)]
    # 蒙版边界比真实物体边缘偏左 3 像素，复现“羽化带跨过轮廓”。
    mask = [[1.0 if x >= w // 2 - 3 else 0.0 for x in range(w)] for _ in range(h)]
    return guide, mask


def test_guided_feather_reduces_halo_and_aligns_edge():
    guide, mask = _offset_scene()
    out = mod.guided_feather(mask, guide, radius=3, eps=1e-3)
    assert len(out) == len(mask) and len(out[0]) == len(mask[0])
    assert out.guided is True
    gplane = mod.as_plane(guide)
    assert out.halo_after < out.halo_before
    plain = mod.gaussian_blur(mask, 3)
    assert mod._halo_energy(out, gplane) < mod._halo_energy(plain, gplane)
    # 引导结果的蒙版梯度更集中在真实引导边缘上（贴合度显著高于普通高斯）。
    assert mod._edge_adherence(out, gplane) > 1.5 * mod._edge_adherence(plain, gplane)
    # 远离轮廓的背景列不应残留可见权重。
    col = lambda x: sum(out[y][x] for y in range(len(out))) / len(out)
    assert col(len(out[0]) // 2 - 9) < 0.05
    # 在蒙版边界到真实边缘之间，引导把权重朝真实边缘重新分布：
    # 真实边缘列应明显比偏左的错误边界内侧列更接近 1。
    w = len(out[0])
    mid = len(out) // 2
    assert out[mid][w // 2] > out[mid][w // 2 - 2]


def test_guided_filter_uses_guide_first_order():
    guide, mask = _offset_scene(24, 18)
    out = mod.guided_filter(guide, mask, radius=2)
    assert len(out) == 18 and len(out[0]) == 24
    assert mod.guided_feather(1) is not None


# --------------------------------------------------------------------------- #
# 修复 3：缓存键带内容版本，ID 不变内容变更时旧结果不命中
# --------------------------------------------------------------------------- #
def test_cache_key_convention():
    assert mod.cache_version("m", "v1") == "m:v1"
    assert mod.cache_version(3, 2) == "3:2"
    assert mod.cache_version("L") == "L"
    assert mod.cache_version(1) is not None


def test_versioned_cache_invalidates_on_content_change():
    cache = mod.VersionedMaskCache()
    cache.put("layer", 1, "old_feather")
    assert cache.get("layer", 1) == "old_feather"
    cache.put("layer", 2, "new_feather")  # ID 相同，版本推进
    assert cache.get("layer", 2) == "new_feather"
    # 旧版本仍是旧结果，但合成只取新版本键，绝不会误命中。
    assert cache.get("layer", 1) == "old_feather"
    assert cache.invalidate("layer") == 2
    assert cache.get("layer", 1) is None and cache.get("layer", 2) is None


def test_feather_uses_versioned_cache():
    mod.mask_cache.clear()
    m = [[0.0, 1.0]] * 4
    r1 = mod.feather(m, layer_id="L9", version=1, radius=1)
    r2 = mod.feather(m, layer_id="L9", version=1, radius=1)
    assert r2.cache_hit is True and r1.cache_key == "L9:1"
    r3 = mod.feather([[1.0, 0.0]] * 4, layer_id="L9", version=2, radius=1)
    assert r3.cache_hit is False and r3.cache_key == "L9:2"


# --------------------------------------------------------------------------- #
# 修复 4：合成前等待正确版本，乱序返回的旧任务不能上屏
# --------------------------------------------------------------------------- #
def test_wait_task_picks_requested_version_data_driven():
    tasks = [{"version": 2, "result": "old"}, {"version": 3, "result": "new"}]
    latest = mod.wait_task(tasks, 3)
    assert latest.ok and latest.result == "new" and latest.fresh is True
    old = mod.wait_task([(2, "old"), (3, "new")], 2)
    assert old.ok and old.result == "old" and old.stale is True and old.fresh is False
    missing = mod.wait_task([{"version": 1, "result": "x"}], 9)
    assert missing.ok is False and missing.result is None
    assert mod.wait_task(1) is not None


def test_stale_predicates():
    assert mod.is_stale(2, 5) is True
    assert mod.is_stale(5, 2) is False
    assert mod.is_fresh({"status": "done", "version": 3}, 3) is True
    assert mod.is_fresh({"status": "pending", "version": 3}, 3) is False


def test_out_of_order_async_completion_keeps_new_version():
    reg = mod.FeatherTaskRegistry()
    reg.submit("L", 1)
    reg.submit("L", 2)
    box = {}

    def waiter():
        box["res"] = reg.wait("L", 2, timeout=5)

    t = threading.Thread(target=waiter, daemon=True)
    t.start()
    reg.complete("L", 1, "OLD")   # 旧版本先（乱序）完成
    time.sleep(0.05)
    assert t.is_alive()           # 新版本未完成，合成确实在等待
    reg.complete("L", 2, "NEW")
    t.join(2.0)
    assert not t.is_alive()
    assert box["res"].ready and box["res"].result == "NEW"
    time.sleep(0.02)
    assert reg.current("L") == "NEW"   # 旧结果没有覆盖当前版本


# --------------------------------------------------------------------------- #
# 修复 5：叠加前归一化，总权重不超过 1
# --------------------------------------------------------------------------- #
def test_normalize_weight_vectors_match_convention():
    assert mod.normalize_overlay([2.0, 2.0]) == pytest.approx([0.5, 0.5])
    assert mod.normalize_overlay([0.0, 0.0]) == [0.5, 0.5]
    assert mod.normalize_overlay([1.0, 0.2]) == pytest.approx([1 / 1.2, 0.2 / 1.2])
    assert sum(mod.normalize_overlay([0.5, 0.5, 0.5])) == pytest.approx(1.0)


def test_multilayer_overlay_never_exceeds_one_and_keeps_sparse():
    a = [[0.8, 0.4, 0.0]]
    b = [[0.7, 0.2, 0.0]]
    out = mod.normalize_overlay([a, b])
    assert isinstance(out, list) and len(out) == 2
    # 单层/稀疏区域不被抬升。
    single = mod.normalize_overlay([a])
    assert single[0][0][0] == pytest.approx(0.8)
    # 重叠超 1 被压缩到和为 1；未超 1 的像素保持原值；全 0 保持 0。
    assert out[0][0][0] + out[1][0][0] == pytest.approx(1.0)
    assert out[0][0][1] + out[1][0][1] == pytest.approx(0.6)
    assert out[0][0][2] + out[1][0][2] == pytest.approx(0.0)
    assert out.max_sum <= 1.0 + 1e-12
    assert mod.normalize_overlay(1) is not None


def test_composite_output_alpha_within_bounds():
    m1 = [[0.9] * 6 for _ in range(4)]
    m2 = [[0.8] * 6 for _ in range(4)]
    rec = mod.composite([m1, m2], background=(0, 0, 0))
    assert rec.ok
    for y in range(rec.height):
        for x in range(rec.width):
            assert 0.0 <= rec.alpha[y][x] <= 1.0
            px = rec.pixels[y][x]
            assert all(0.0 <= c <= 1.0 for c in px)


# --------------------------------------------------------------------------- #
# 修复 6：边缘 / 权重钳制 + replicate 扩边
# --------------------------------------------------------------------------- #
def test_clamp_scalar_and_index():
    assert mod.clamp_edge(2.0) == 1.0
    assert mod.clamp_edge(-0.3) == 0.0
    assert mod.clamp_edge(1) == 1.0
    assert mod.clamp_edge(9, 5) == 4.0
    assert mod.clamp_edge(-2, 5) == 0.0
    assert mod.clamp_edge(0.5, 0.0, 1.0) == pytest.approx(0.5)


def test_clamp_grid_and_replicate_pad():
    grid = [[-1.0, 0.5], [0.2, 2.0]]
    out = mod.clamp_edge(grid)
    assert out[0][0] == 0.0 and out[1][1] == 1.0 and out[0][1] == pytest.approx(0.5)
    padded = mod.clamp_edge([[1.0, 0.0]], pad=1).padded
    assert len(padded) == 3 and len(padded[0]) == 4
    # 四角 replicate：右下角等于真实右下角 0。
    assert padded[2][3] == 0.0 and padded[0][0] == 1.0


# --------------------------------------------------------------------------- #
# 管线 load / feather / composite / save
# --------------------------------------------------------------------------- #
def test_feather_pipeline_scales_and_clamps():
    mask = [[0.0] * 12 + [1.0] * 12 for _ in range(10)]
    out = mod.feather(mask, radius=2, resolution=2160, guided=False)
    assert len(out) == 10 and len(out[0]) == 24
    assert out.radius == pytest.approx(4.0)  # 2 * 2160/1080
    assert all(0.0 <= out[y][x] <= 1.0 for y in range(10) for x in range(24))


def test_save_ppm_and_load_roundtrip(tmp_path):
    rec = mod.save(1)
    assert rec.ok and rec.magic == "P3" and rec.text.startswith("P3")
    ppm = tmp_path / "out.ppm"
    saved = mod.save(mod.synthetic_scene(8, 6), str(ppm))
    assert saved.ok and ppm.exists() and saved.bytes_written > 0
    loaded = mod.load(str(ppm))
    assert loaded.ok and loaded.width == 8 and loaded.height == 6


def test_save_json_record(tmp_path):
    jp = tmp_path / "audit.json"
    rec = mod.save(mod.audit(), str(jp), format="json")
    assert rec.ok and jp.exists()
    json.loads(jp.read_text(encoding="utf-8"))


def test_load_missing_returns_error_record_not_raise(tmp_path):
    rec = mod.load(str(tmp_path / "nope.ppm"))
    assert rec.ok is False and rec.fallback is not None


# --------------------------------------------------------------------------- #
# validate / metrics / audit / fallback / determinism
# --------------------------------------------------------------------------- #
def test_validate_mask():
    assert mod.validate_mask([[0.0, 1.0], [0.5, 0.5]]).ok is True
    bad = mod.validate_mask(object())
    assert bad.ok is False and bad.problems
    assert mod.validate_mask(None).ok is False


def test_metrics_shapes_and_garbage():
    demo = mod.metrics(1)
    assert demo.ok and demo.feather_width_px >= 0
    assert mod.metrics(object()).ok is False


def test_audit_real_counters_and_zero_stubs():
    aud = mod.Auditor()
    aud.incr("tiles", 3)
    aud.event("run", "x")
    rep = aud.report()
    assert rep.counters["tiles"] == 3 and rep.counters["run"] == 1
    aud.reset()
    assert aud.report().counters == {}
    top = mod.audit()
    assert top.ok and top.stubs_remaining == 0 and top.all_implemented is True


def test_fallback_shape():
    res = mod.fallback(object())
    assert res.ok and res.degraded and res.finite
    assert len(res.image) == 12 and len(res.image[0]) == 16


def test_determinism_exact():
    res = mod.determinism()
    assert res.deterministic is True
    assert res.max_abs_diff == 0.0
    assert res.same_param_id is True and res.runs == 2


# --------------------------------------------------------------------------- #
# 全部入口 fn(1) 非 None 且可 JSON 序列化；顶层不依赖 numpy/cv2/PIL
# --------------------------------------------------------------------------- #
NAMES = ["radius_by_resolution", "guided_feather", "cache_version", "wait_task",
         "normalize_overlay", "clamp_edge", "load", "feather", "composite",
         "save", "validate_mask", "metrics", "audit", "fallback", "determinism"]


@pytest.mark.parametrize("name", NAMES)
def test_every_entry_fn1_json_safe(name):
    result = getattr(mod, name)(1)
    assert result is not None
    json.dumps(result, default=str)


def test_no_top_level_third_party_imports():
    src = open(mod.__file__, encoding="utf-8").read()
    tree = ast.parse(src)
    tops = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            tops.extend(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            tops.append(node.module.split(".")[0])
    assert "numpy" not in tops and "cv2" not in tops and "PIL" not in tops
