/**
 * Drag-and-drop file upload zone component.
 */

const ACCEPTED_IMAGE_TYPES = [
  "image/jpeg",
  "image/png",
  "image/bmp",
  "image/webp",
];
const ACCEPTED_VIDEO_TYPES = ["video/mp4", "video/avi", "video/quicktime"];
const ALL_ACCEPTED = [...ACCEPTED_IMAGE_TYPES, ...ACCEPTED_VIDEO_TYPES];

/**
 * Create the upload zone component.
 * @param {function} onFileSelected - Callback when a file is selected.
 * @returns {HTMLElement}
 */
export function createUploadZone(onFileSelected) {
  const container = document.createElement("div");
  container.innerHTML = `
    <div class="upload-zone" id="upload-drop-zone">
      <input type="file" id="file-input" class="hidden"
             accept=".jpg,.jpeg,.png,.bmp,.webp,.mp4,.avi,.mov" />

      <div class="flex flex-col items-center gap-4">
        <!-- Upload icon -->
        <div class="w-20 h-20 rounded-2xl bg-bg-tertiary border border-border flex items-center justify-center transition-all duration-300" id="upload-icon-box">
          <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" class="text-accent-light">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
            <polyline points="17 8 12 3 7 8"/>
            <line x1="12" y1="3" x2="12" y2="15"/>
          </svg>
        </div>

        <div>
          <p class="text-text-primary font-semibold text-lg mb-1">
            Drop your file here or <span class="text-accent-light cursor-pointer hover:underline" id="browse-trigger">browse</span>
          </p>
          <p class="text-text-muted text-sm">
            Supports JPG, PNG, MP4 • Max 50MB images, 200MB video
          </p>
        </div>
      </div>

      <!-- File preview (hidden initially) -->
      <div id="file-preview" class="hidden mt-6">
        <div class="glass-card p-4 inline-flex items-center gap-4 max-w-md mx-auto">
          <div id="preview-thumbnail" class="w-16 h-16 rounded-lg bg-bg-tertiary overflow-hidden flex-shrink-0 flex items-center justify-center">
          </div>
          <div class="text-left flex-1 min-w-0">
            <p class="text-text-primary font-medium text-sm truncate" id="preview-filename"></p>
            <p class="text-text-muted text-xs" id="preview-filesize"></p>
          </div>
          <button id="preview-remove" class="w-8 h-8 rounded-lg bg-bg-tertiary hover:bg-accent/20 flex items-center justify-center text-text-muted hover:text-ember transition-colors flex-shrink-0">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
            </svg>
          </button>
        </div>
      </div>
    </div>
  `;

  const zone = container.querySelector("#upload-drop-zone");
  const fileInput = container.querySelector("#file-input");
  const browseTrigger = container.querySelector("#browse-trigger");
  const previewEl = container.querySelector("#file-preview");
  const previewThumb = container.querySelector("#preview-thumbnail");
  const previewName = container.querySelector("#preview-filename");
  const previewSize = container.querySelector("#preview-filesize");
  const removeBtn = container.querySelector("#preview-remove");

  let selectedFile = null;

  // Click to browse
  browseTrigger.addEventListener("click", (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  zone.addEventListener("click", (e) => {
    if (e.target.id !== "preview-remove" && !e.target.closest("#preview-remove")) {
      fileInput.click();
    }
  });

  // Drag events
  zone.addEventListener("dragover", (e) => {
    e.preventDefault();
    zone.classList.add("drag-over");
  });

  zone.addEventListener("dragleave", () => {
    zone.classList.remove("drag-over");
  });

  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });

  // File input change
  fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (file) handleFile(file);
  });

  // Remove file
  removeBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    clearFile();
  });

  function handleFile(file) {
    if (!ALL_ACCEPTED.includes(file.type)) {
      alert("Unsupported file type. Please use JPG, PNG, or MP4.");
      return;
    }

    const maxSize = ACCEPTED_VIDEO_TYPES.includes(file.type)
      ? 200 * 1024 * 1024
      : 50 * 1024 * 1024;

    if (file.size > maxSize) {
      alert(`File too large. Max: ${maxSize / (1024 * 1024)} MB`);
      return;
    }

    selectedFile = file;

    // Show preview
    previewName.textContent = file.name;
    previewSize.textContent = formatSize(file.size);
    previewThumb.innerHTML = "";

    if (ACCEPTED_IMAGE_TYPES.includes(file.type)) {
      const img = document.createElement("img");
      img.src = URL.createObjectURL(file);
      img.className = "w-full h-full object-cover";
      previewThumb.appendChild(img);
    } else {
      previewThumb.innerHTML = `
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" class="text-smoke">
          <polygon points="5 3 19 12 5 21 5 3"/>
        </svg>
      `;
    }

    previewEl.classList.remove("hidden");
    onFileSelected(file);
  }

  function clearFile() {
    selectedFile = null;
    fileInput.value = "";
    previewEl.classList.add("hidden");
    previewThumb.innerHTML = "";
    onFileSelected(null);
  }

  function formatSize(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  }

  return container.firstElementChild.parentElement
    ? container
    : container;
}
