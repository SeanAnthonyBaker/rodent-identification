import asyncio
import base64
import json
import time
from pathlib import Path
import httpx

images = [
    "C:/Users/seanb/.gemini/antigravity/brain/af808bf8-12fa-4824-a721-6fa73449fa13/.user_uploaded/media_1787658777246.png",
    "C:/Users/seanb/.gemini/antigravity/brain/af808bf8-12fa-4824-a721-6fa73449fa13/.user_uploaded/media_1787658778109.png",
    "C:/Users/seanb/.gemini/antigravity/brain/af808bf8-12fa-4824-a721-6fa73449fa13/.user_uploaded/media_1787658778702.png",
    "C:/Users/seanb/.gemini/antigravity/brain/af808bf8-12fa-4824-a721-6fa73449fa13/.user_uploaded/media_1787658797663.png",
    "C:/Users/seanb/.gemini/antigravity/brain/af808bf8-12fa-4824-a721-6fa73449fa13/.user_uploaded/media_1787658816489.png"
]

async def evaluate_models():
    candidate_models = ["tulkah_gemma4_12b:latest", "gemma3:12b", "gemma3:4b"]
    
    prompt = (
        "Carefully analyze this image for wildlife and pest detection. "
        "Is there a rat, mouse, or rodent present in this image? "
        "Answer strictly in valid JSON format:\n"
        "{\n"
        '  "rat_detected": true or false,\n'
        '  "confidence": float between 0.0 and 1.0,\n'
        '  "description": "short explanation of observation",\n'
        '  "bounding_box": [ymin, xmin, ymax, xmax] or null\n'
        "}"
    )

    async with httpx.AsyncClient(timeout=120.0) as client:
        # Check which models respond
        resp = await client.get("http://localhost:11434/api/tags")
        installed = [m["name"] for m in resp.json().get("models", [])]
        print(f"Installed models in Ollama: {installed}\n")

        model_to_use = "tulkah_gemma4_12b:latest"
        if model_to_use not in installed and "hf.co/unsloth/gemma-4-12b-it-GGUF:Q4_K_M" in installed:
            model_to_use = "hf.co/unsloth/gemma-4-12b-it-GGUF:Q4_K_M"

        print(f"=== BENCHMARKING MODEL: {model_to_use} ===\n")
        
        results = []
        for idx, img_path in enumerate(images, 1):
            with open(img_path, "rb") as f:
                img_bytes = f.read()
                b64 = base64.b64encode(img_bytes).decode("utf-8")

            t0 = time.perf_counter()
            try:
                r = await client.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": model_to_use,
                        "prompt": prompt,
                        "images": [b64],
                        "stream": False,
                        "format": "json"
                    }
                )
                elapsed = (time.perf_counter() - t0) * 1000
                res_data = r.json()
                raw_text = res_data.get("response", "")
                parsed = json.loads(raw_text) if raw_text else {}
                
                results.append({
                    "image_index": idx,
                    "filename": Path(img_path).name,
                    "latency_ms": round(elapsed, 1),
                    "rat_detected": parsed.get("rat_detected"),
                    "confidence": parsed.get("confidence"),
                    "description": parsed.get("description"),
                    "bounding_box": parsed.get("bounding_box"),
                    "raw": raw_text
                })
                print(f"[{idx}/5] {Path(img_path).name} -> Rat Detected: {parsed.get('rat_detected')} (Conf: {parsed.get('confidence')}) [{elapsed:.0f}ms]")
                print(f"     Description: {parsed.get('description')}\n")

            except Exception as e:
                print(f"[{idx}/5] Error on {Path(img_path).name}: {e}")

        with open("benchmark_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print("Completed! Saved to benchmark_results.json")

if __name__ == "__main__":
    asyncio.run(evaluate_models())
