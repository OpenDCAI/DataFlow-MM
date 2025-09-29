import base64
import random
from io import BytesIO
from typing import List
from PIL import Image

def combine_images_to_base64(image_paths: List[str], mode: str = "combine") -> str:
    """
    Simple standalone version to test the 'combine' mode.
    """
    images = [Image.open(path) for path in image_paths]

    if mode == "combine":
        # === 改为网格铺放（允许留白、不裁剪），避免重叠 ===
        max_dim = max(max(img.width, img.height) for img in images)
        canvas_size = max(1024, max_dim)
        combined_image = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))

        n = len(images)
        import math
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)

        def _partition_sizes(total: int, parts: int):
            base = total // parts
            rem = total - base * parts
            return [base + 1 if i < rem else base for i in range(parts)]

        # 等比缩放以“完全装入”目标格子（不裁剪）
        def _fit_resize(img: Image.Image, tw: int, th: int) -> Image.Image:
            w, h = img.size
            if w == 0 or h == 0:  # 防御
                return Image.new("RGBA", (tw, th), (255, 255, 255, 0))
            scale = min(tw / w, th / h)
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            return img.convert("RGBA").resize((nw, nh), Image.Resampling.LANCZOS)

        row_heights = _partition_sizes(canvas_size, rows)
        last_row_cols = n - (rows - 1) * cols
        if last_row_cols <= 0:
            last_row_cols = cols

        idx = 0
        y = 0
        for r in range(rows):
            h_r = row_heights[r]
            cols_r = cols if (r < rows - 1) else last_row_cols
            col_widths = _partition_sizes(canvas_size, cols_r)
            x = 0
            for c in range(cols_r):
                if idx >= n:
                    break
                w_c = col_widths[c]
                tile = _fit_resize(images[idx], w_c, h_r)
                # 居中贴到该格子内（允许四周留白）
                ox = x + (w_c - tile.width) // 2
                oy = y + (h_r - tile.height) // 2
                combined_image.paste(
                    tile,
                    (ox, oy),
                    mask=tile.split()[3] if tile.mode == "RGBA" else None
                )
                x += w_c
                idx += 1
            y += h_r

    else:
        raise ValueError("Only 'combine' mode is supported in this test.")

    # 输出 base64
    buffer = BytesIO()
    combined_image.save(buffer, format="PNG")
    base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return base64_image


if __name__ == "__main__":
    # 替换成本地的几张测试图片路径
    # test_images = ["./cache_local/condition_images/Matte_black_wireless_earbuds_on_a_marble_surface_soft_shadow_studio_lighting_high_contrast/Matte_black_wireless_earbuds_on_a_marble_surface_soft_shadow_studio_lighting_high_contrast_0.png","./cache_local/condition_images/Smartwatch_floating_exploded_view_showing_components_isometric_white_background_crisp_lines/Smartwatch_floating_exploded_view_showing_components_isometric_white_background_crisp_lines_0.png","./cache_local/condition_images/Eco_friendly_shampoo_bottle_with_dew_drops_on_moss_morning_light_macro_lens/Eco_friendly_shampoo_bottle_with_dew_drops_on_moss_morning_light_macro_lens_0.png","./cache_local/condition_images/Sneakers_mid_air_with_motion_blur_trails_colorful_gradient_backdrop_commercial_style/Sneakers_mid_air_with_motion_blur_trails_colorful_gradient_backdrop_commercial_style_0.png"]


    # test_images = ["./cache_local/condition_images/A_serene_lakeside_at_sunrise_soft_mist_over_thewater_cinematic_lighting_35mm_high_detail/A_serene_lakeside_at_sunrise_soft_mist_over_thewater_cinematic_lighting_35mm_high_detail_0.png","./cache_local/condition_images/Cyberpunk_street_at_night_neon_signs_rain_soaked_pavement_reflections_wide_angle_Octane_render/Cyberpunk_street_at_night_neon_signs_rain_soaked_pavement_reflections_wide_angle_Octane_render_0.png","./cache_local/condition_images/Cozy_Scandinavian_living_room_natural_light_minimal_decor_warm_tones_photorealistic/Cozy_Scandinavian_living_room_natural_light_minimal_decor_warm_tones_photorealistic_0.png"]

    test_images = ["/m2v_intern/wangyuran/dataflow/demo/bottle_white_bg.png", "/m2v_intern/wangyuran/dataflow/demo/earphone_white_bg.png", "/m2v_intern/wangyuran/dataflow/demo/sneaker_white_bg.png", "/m2v_intern/wangyuran/dataflow/demo/watch_white_bg.png"]

    b64 = combine_images_to_base64(test_images, mode="combine")
    print("Base64 length:", len(b64))

    img_bytes = base64.b64decode(b64)
    with open("combined_result.png", "wb") as f:
        f.write(img_bytes)
    print("Saved combined_result.png")
