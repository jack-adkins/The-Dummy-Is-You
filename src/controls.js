// processes the user/client's actions, computes deltaXY which are then used by camera 
export class Controls {
    constructor(renderer) {
        this.renderer = renderer;
        this.canvas = renderer.canvas;
        this.camera = renderer.camera;
        this.isectOnly = false;     // intersection only mode
        this.isDragging = false;    // isDragging is when user is holding down mouse + moving it
        this.prevMouseX = 0;        // where the mouse was last seen 
        this.prevMouseY = 0;        // where the mouse was last seen 
        this.setupEventListeners(); // listen to client event 
    }

    setupEventListeners() {
        // Max Depth slider
        const recursionSlider = document.getElementById("maxDepth");
        recursionSlider.addEventListener("input", (event) => {
            this.renderer.maxDepth = parseInt(event.target.value);
            document.getElementById("maxDepthVal").innerText = event.target.value;
        });
    }

    updateCameraInfo() {
        // Camera info update no longer needed since we removed the sliders
        // This method can be kept for future use or removed entirely
    }
}
