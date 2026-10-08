from __future__ import annotations

from PIL import ImageDraw


class Localiser:
    """Conservative coarse matching followed by native-resolution refinement."""

    def __init__(self, config):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Localisation requires the [vlm] extra (headless OpenCV)") from exc
        self.cv2 = cv2
        self.config = config
        self.previous = {}

    def locate(self, crop, global_image, track_key: str) -> dict:
        if crop is None or global_image is None:
            return {"status": "unavailable", "reason": "source_missing"}
        import numpy as np

        cv2 = self.cv2
        scene = cv2.cvtColor(np.asarray(global_image), cv2.COLOR_RGB2GRAY)
        target = cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2GRAY)
        if float(target.std()) < 2:
            return {"status": "rejected", "reason": "low_texture"}
        factor = min(1.0, 960 / scene.shape[1])
        coarse = cv2.resize(scene, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        candidates = []
        for scale in self.config.localisation_scales:
            width, height = round(target.shape[1] * scale), round(target.shape[0] * scale)
            cw, ch = max(2, round(width * factor)), max(2, round(height * factor))
            if cw > coarse.shape[1] or ch > coarse.shape[0] or min(width, height) < 2:
                continue
            template = cv2.resize(target, (cw, ch), interpolation=cv2.INTER_AREA)
            if float(template.std()) < 2:
                continue
            response = cv2.matchTemplate(coarse, template, cv2.TM_CCOEFF_NORMED)
            if not np.isfinite(response).all():
                continue
            # Inspect a previous neighbourhood first, but still measure scene-wide alternatives.
            previous = self.previous.get(track_key)
            if previous:
                px, py = round(previous[0] * factor), round(previous[1] * factor)
                radius = max(cw, ch)
                region = response[
                    max(0, py - radius) : py + radius + 1, max(0, px - radius) : px + radius + 1
                ]
                if region.size:
                    _, _, _, local = cv2.minMaxLoc(region)
                    x = local[0] + max(0, px - radius)
                    y = local[1] + max(0, py - radius)
                    candidates.append((float(response[y, x]), x, y, width, height))
            for _ in range(2):
                _, score, _, (x, y) = cv2.minMaxLoc(response)
                candidates.append((float(score), x, y, width, height))
                # Distinct alternatives have substantially different centres.
                response[max(0, y - ch // 2) : y + ch // 2 + 1, max(0, x - cw // 2) : x + cw // 2 + 1] = -1
        if not candidates:
            return {"status": "rejected", "reason": "no_valid_template_scale"}
        refined = []
        for _, x, y, width, height in candidates:
            nx, ny = round(x / factor), round(y / factor)
            radius = max(4, round(4 / factor))
            left, top = max(0, nx - radius), max(0, ny - radius)
            right, bottom = (
                min(scene.shape[1], nx + width + radius),
                min(scene.shape[0], ny + height + radius),
            )
            template = cv2.resize(target, (width, height), interpolation=cv2.INTER_LINEAR)
            region = scene[top:bottom, left:right]
            if region.shape[0] < height or region.shape[1] < width:
                continue
            response = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
            _, score, _, (rx, ry) = cv2.minMaxLoc(response)
            if np.isfinite(score):
                refined.append((float(score), [left + rx, top + ry, left + rx + width, top + ry + height]))
        if not refined:
            return {"status": "rejected", "reason": "refinement_out_of_bounds"}
        refined.sort(key=lambda item: item[0], reverse=True)
        score, bbox = refined[0]
        width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        alternatives = [
            s
            for s, other in refined[1:]
            if abs((other[0] + other[2]) / 2 - (bbox[0] + bbox[2]) / 2) > min(other[2] - other[0], width) / 2
            or abs((other[1] + other[3]) / 2 - (bbox[1] + bbox[3]) / 2) > min(other[3] - other[1], height) / 2
        ]
        alternative = max(alternatives, default=-1.0)
        margin = float(score) - alternative
        accepted = (
            float(score) >= self.config.localisation_score and margin >= self.config.localisation_margin
        )
        reason = None if accepted else "score_or_distinct_match_margin"
        previous = self.previous.get(track_key)
        if (
            accepted
            and previous
            and (abs(bbox[0] - previous[0]) + abs(bbox[1] - previous[1]) > 2 * max(width, height))
        ):
            reason = "large_target_jump_review"
        if accepted:
            self.previous[track_key] = bbox
        return {
            "status": "accepted" if accepted else "rejected",
            "bbox": bbox if accepted else None,
            "score": float(score),
            "margin": float(margin),
            "reason": reason,
        }


def highlight(image, location):
    if image is None or location["status"] != "accepted":
        return image
    temporary = image.copy()
    ImageDraw.Draw(temporary).rectangle(location["bbox"], outline=(255, 0, 0), width=4)
    return temporary
