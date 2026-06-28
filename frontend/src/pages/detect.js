/**
 * Detection page — Upload image/video, run ONNX inference, display results.
 */

import { createUploadZone } from "../components/upload.js";

// API base URL — in dev, Vite proxies /api to localhost:8080
// In production, set this to the Cloud Run URL
const API_BASE = import.meta.env.VITE_API_URL || "";

export function renderDetect(container) {
  container.innerHTML = "";
  container.className = "page-container page-enter";

  let selectedFile = null;

  const page = document.createElement("div");
  page.className = "max-w-6xl mx-auto px-6 pt-28 pb-20";

  // Header
  page.innerHTML = `
    <div class="text-center mb-10">
      <h1 class="font-display font-bold text-4xl md:text-5xl mb-3">
        <span class="gradient-text">Wildfire Detection</span>
      </h1>
      <p class="text-text-secondary text-lg max-w-xl mx-auto">
        Upload an image or video and our YOLOv11 model will detect early signs of fire and smoke.
      </p>
    </div>

    <!-- Main content area -->
    <div class="grid lg:grid-cols-2 gap-6">
      <!-- Left: Upload + Controls -->
      <div>
        <div id="upload-container"></div>

        <!-- Controls -->
        <div class="glass-card p-6 mt-4">
          <div class="flex items-center justify-between mb-4">
            <label class="text-sm font-semibold text-text-primary">Confidence Threshold</label>
            <span class="text-sm font-mono text-accent-light" id="conf-value">0.38</span>
          </div>
          <input type="range" id="conf-slider" min="0.01" max="0.90" step="0.01" value="0.38"
                 class="w-full h-2 rounded-full appearance-none cursor-pointer
                        bg-bg-tertiary accent-accent-light" />
          <div class="flex justify-between text-xs text-text-muted mt-1">
            <span>0.01</span>
            <span>Sensitive ← → Strict</span>
            <span>0.90</span>
          </div>

          <button id="detect-btn"
                  class="btn-accent w-full mt-6 py-4 text-lg flex items-center justify-center gap-3"
                  disabled>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
            </svg>
            <span id="detect-btn-text">Run Detection</span>
          </button>
        </div>
      </div>

      <!-- Right: Results -->
      <div>
        <div class="glass-card p-6 min-h-[400px] flex flex-col" id="results-panel">
          <!-- Empty state -->
          <div id="results-empty" class="flex-1 flex flex-col items-center justify-center text-center py-10">
            <div class="w-20 h-20 rounded-2xl bg-bg-tertiary border border-border flex items-center justify-center mb-4">
              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" class="text-text-muted">
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
                <circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/>
              </svg>
            </div>
            <p class="text-text-muted font-medium">Detection results will appear here</p>
            <p class="text-text-muted text-sm mt-1">Upload an image or video to get started</p>
          </div>

          <!-- Loading state -->
          <div id="results-loading" class="hidden flex-1 flex flex-col items-center justify-center text-center py-10">
            <div class="loading-spinner mb-4"></div>
            <p class="text-text-primary font-medium" id="loading-text">Processing...</p>
            <p class="text-text-muted text-sm mt-1">Running ONNX inference on your file</p>
          </div>

          <!-- Image result -->
          <div id="results-image" class="hidden flex-1 flex flex-col">
            <div class="flex items-center justify-between mb-3">
              <h3 class="font-semibold text-text-primary">Detection Result</h3>
              <div class="flex items-center gap-2">
                <span class="text-xs text-text-muted font-mono" id="result-time"></span>
                <button id="download-btn"
                        class="px-3 py-1.5 rounded-lg bg-bg-tertiary border border-border text-sm text-text-secondary hover:text-text-primary hover:border-accent/50 transition-all flex items-center gap-1.5">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>
                  </svg>
                  Download
                </button>
              </div>
            </div>
            <div class="flex-1 rounded-xl bg-bg-tertiary mb-3 relative flex items-center justify-center overflow-hidden" style="min-height: 300px;">
              <div class="relative inline-block max-w-full" id="result-image-container">
                <img id="result-img" class="max-w-full max-h-[500px] block" src="" alt="Detection result" />
              </div>
            </div>
            <!-- Detection list -->
            <div id="detection-list" class="space-y-2 max-h-48 overflow-y-auto"></div>
          </div>

          <!-- Video result -->
          <div id="results-video" class="hidden flex-1 flex flex-col">
            <div class="flex items-center justify-between mb-3">
              <h3 class="font-semibold text-text-primary">Video Result</h3>
              <button id="download-video-btn"
                      class="px-3 py-1.5 rounded-lg bg-bg-tertiary border border-border text-sm text-text-secondary hover:text-text-primary hover:border-accent/50 transition-all flex items-center gap-1.5">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>
                </svg>
                Download
              </button>
            </div>
            <div class="flex-1 rounded-xl overflow-hidden bg-bg-tertiary">
              <video id="result-video" controls class="w-full h-full object-contain"></video>
            </div>
          </div>

          <!-- Error state -->
          <div id="results-error" class="hidden flex-1 flex flex-col items-center justify-center text-center py-10">
            <div class="w-16 h-16 rounded-full bg-accent/20 flex items-center justify-center mb-4">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="text-ember">
                <circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>
              </svg>
            </div>
            <p class="text-ember font-medium" id="error-message">An error occurred</p>
            <p class="text-text-muted text-sm mt-1">Please try again with a different file</p>
          </div>
        </div>
      </div>
    </div>
  `;

  container.appendChild(page);

  // --- Setup upload zone ---
  const uploadContainer = page.querySelector("#upload-container");
  const uploadZone = createUploadZone((file) => {
    selectedFile = file;
    const detectBtn = page.querySelector("#detect-btn");
    detectBtn.disabled = !file;
    // Reset results when new file selected
    showPanel("results-empty");
  });
  uploadContainer.appendChild(uploadZone);

  // --- Confidence slider ---
  const confSlider = page.querySelector("#conf-slider");
  const confValue = page.querySelector("#conf-value");
  confSlider.addEventListener("input", () => {
    confValue.textContent = parseFloat(confSlider.value).toFixed(2);
  });

  // --- Detect button ---
  const detectBtn = page.querySelector("#detect-btn");
  detectBtn.addEventListener("click", () => {
    if (!selectedFile) return;
    runDetection(selectedFile, parseFloat(confSlider.value), page);
  });

  // --- Download button ---
  page.querySelector("#download-btn").addEventListener("click", () => {
    const img = page.querySelector("#result-img");
    if (img.src) {
      const a = document.createElement("a");
      a.href = img.src;
      a.download = "detected_result.jpg";
      a.click();
    }
  });

  function showPanel(panelId) {
    [
      "results-empty",
      "results-loading",
      "results-image",
      "results-video",
      "results-error",
    ].forEach((id) => {
      page.querySelector(`#${id}`).classList.toggle("hidden", id !== panelId);
    });
  }

  async function runDetection(file, confidence, page) {
    showPanel("results-loading");
    const detectBtnText = page.querySelector("#detect-btn-text");
    const detectBtn = page.querySelector("#detect-btn");
    detectBtn.disabled = true;
    detectBtnText.textContent = "Processing...";

    const isVideo = file.type.startsWith("video/");
    const loadingText = page.querySelector("#loading-text");
    loadingText.textContent = isVideo
      ? "Processing video frames..."
      : "Running detection...";

    const formData = new FormData();
    formData.append("file", file);
    formData.append("confidence", confidence);

    try {
      const endpoint = isVideo ? "/api/detect/video" : "/api/detect/image";
      const response = await fetch(`${API_BASE}${endpoint}`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.error || `Server error: ${response.status}`);
      }

      if (isVideo) {
        // Video response is a file
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const video = page.querySelector("#result-video");
        video.src = url;
        showPanel("results-video");

        // Download handler
        page.querySelector("#download-video-btn").onclick = () => {
          const a = document.createElement("a");
          a.href = url;
          a.download = "detected_output.mp4";
          a.click();
        };
      } else {
        // Image response is JSON
        const data = await response.json();

        // Show annotated image
        const resultImg = page.querySelector("#result-img");
        resultImg.src = `data:image/jpeg;base64,${data.annotated_image}`;

        // Processing time
        page.querySelector("#result-time").textContent =
          `${data.processing_time}s`;

        // Add interactive hover overlays
        const container = page.querySelector("#result-image-container");
        // Remove old overlays
        container.querySelectorAll(".bbox-overlay").forEach(el => el.remove());

        const imgWidth = data.image_size.width;
        const imgHeight = data.image_size.height;

        data.detections.forEach((d) => {
          const [x1, y1, x2, y2] = d.bbox;
          const left = (x1 / imgWidth) * 100;
          const top = (y1 / imgHeight) * 100;
          const w = ((x2 - x1) / imgWidth) * 100;
          const h = ((y2 - y1) / imgHeight) * 100;

          const overlay = document.createElement("div");
          overlay.className = "bbox-overlay absolute group cursor-pointer border-2 border-transparent hover:border-white hover:bg-white/10 transition-all z-10";
          overlay.style.left = `${left}%`;
          overlay.style.top = `${top}%`;
          overlay.style.width = `${w}%`;
          overlay.style.height = `${h}%`;

          const color = d.class === "Early_Fire" ? "#ef4444" : "#f97316";
          overlay.innerHTML = `
            <div class="absolute -top-8 left-1/2 -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity bg-[#1a1313] border border-white/10 text-white px-2 py-1 rounded text-xs whitespace-nowrap shadow-xl pointer-events-none z-20 font-mono">
              <span class="font-bold" style="color: ${color}">${d.class}</span> 
              ${(d.confidence * 100).toFixed(1)}%
            </div>
          `;
          container.appendChild(overlay);
        });

        // Detection list
        const listEl = page.querySelector("#detection-list");
        if (data.detections.length === 0) {
          listEl.innerHTML = `
            <div class="p-3 rounded-lg bg-bg-tertiary text-center text-text-muted text-sm">
              No detections found. Try lowering the confidence threshold.
            </div>
          `;
        } else {
          listEl.innerHTML = data.detections
            .map(
              (d) => `
            <div class="flex items-center gap-3 p-3 rounded-lg bg-bg-tertiary">
              <span class="w-3 h-3 rounded-full flex-shrink-0" style="background: ${d.class === "Early_Fire" ? "#dc2626" : "#f97316"}"></span>
              <span class="text-sm font-medium text-text-primary flex-1">${d.class}</span>
              <span class="text-xs font-mono text-accent-light">${(d.confidence * 100).toFixed(1)}%</span>
              <span class="text-xs font-mono text-text-muted">[${d.bbox.join(", ")}]</span>
            </div>
          `
            )
            .join("");
        }

        showPanel("results-image");
      }
    } catch (err) {
      console.error("Detection error:", err);
      page.querySelector("#error-message").textContent = err.message;
      showPanel("results-error");
    } finally {
      detectBtn.disabled = false;
      detectBtnText.textContent = "Run Detection";
    }
  }
}
