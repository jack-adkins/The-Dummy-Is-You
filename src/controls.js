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
        // Reset button
        const resetButton = document.getElementById("resetButton");
        resetButton.addEventListener("click", (event) => {
            this.renderer.resetScene();
        });
        // Reload Shaders button
        const reloadButton = document.getElementById("reloadButton");
        reloadButton.addEventListener("click", async (event) => {
            this.renderer.reloadShaders("test");
        });
        // Recursion step
        const recursionSlider = document.getElementById("maxDepth");
        recursionSlider.addEventListener("input", (event) => {
            this.renderer.maxDepth = parseInt(event.target.value);
            document.getElementById("maxDepthVal").innerText = event.target.value;
        });
        // Camera rotation
        const rotateUSlider = document.getElementById("rotateU");
        const rotateVSlider = document.getElementById("rotateV");
        const rotateWSlider = document.getElementById("rotateW");
        rotateUSlider.addEventListener("input", (event) => {
            this.camera.setRotUVW(
                parseFloat(event.target.value),
                this.camera.rotV,
                this.camera.rotW
            );
            document.getElementById("rotateUVal").innerText = event.target.value;
        });
        rotateVSlider.addEventListener("input", (event) => {
            this.camera.setRotUVW(
                this.camera.rotU,
                parseFloat(event.target.value),
                this.camera.rotW
            );
            document.getElementById("rotateVVal").innerText = event.target.value;
        });
        rotateWSlider.addEventListener("input", (event) => {
            this.camera.setRotUVW(
                this.camera.rotU,
                this.camera.rotV,
                parseFloat(event.target.value)
            );
            document.getElementById("rotateWVal").innerText = event.target.value;
        });

        // Camera translation
        const eyeXSlider = document.getElementById("eyeX");
        const eyeYSlider = document.getElementById("eyeY");
        const eyeZSlider = document.getElementById("eyeZ");
        eyeXSlider.addEventListener("input", (event) => {
            this.camera.orientLookVec(
                [parseFloat(event.target.value), eyeYSlider.value, eyeZSlider.value],
                this.camera.getLookVector(),
                this.camera.getUpVector()
            )
            document.getElementById("eyeXVal").innerText = event.target.value;
        });
        eyeYSlider.addEventListener("input", (event) => {
            this.camera.orientLookVec(
                [eyeXSlider.value, parseFloat(event.target.value), eyeZSlider.value],
                this.camera.getLookVector(),
                this.camera.getUpVector()
            )
            document.getElementById("eyeYVal").innerText = event.target.value;
        });
        eyeZSlider.addEventListener("input", (event) => {
            this.camera.orientLookVec(
                [eyeXSlider.value, eyeYSlider.value, parseFloat(event.target.value)],
                this.camera.getLookVector(),
                this.camera.getUpVector()
            )
            document.getElementById("eyeZVal").innerText = event.target.value;
        });

        // Look vector
        const lookXSlider = document.getElementById("lookX");
        const lookYSlider = document.getElementById("lookY");
        const lookZSlider = document.getElementById("lookZ");
        lookXSlider.addEventListener("input", (event) => {
            this.camera.orientLookVec(
                this.camera.getEyePoint(),
                [parseFloat(event.target.value), lookYSlider.value, lookZSlider.value],
                this.camera.getUpVector()
            )
            document.getElementById("lookXVal").innerText = event.target.value;
        });
        lookYSlider.addEventListener("input", (event) => {
            this.camera.orientLookVec(
                this.camera.getEyePoint(),
                [lookXSlider.value, parseFloat(event.target.value), lookZSlider.value],
                this.camera.getUpVector()
            )
            document.getElementById("lookYVal").innerText = event.target.value;
        });
        lookZSlider.addEventListener("input", (event) => {
            this.camera.orientLookVec(
                this.camera.getEyePoint(),
                [lookXSlider.value, lookYSlider.value, parseFloat(event.target.value)],
                this.camera.getUpVector()
            )
            document.getElementById("lookZVal").innerText = event.target.value;
        });

        // Near and far planes
        const nearSlider = document.getElementById("near");
        const farSlider = document.getElementById("far");
        nearSlider.addEventListener("input", (event) => {
            this.camera.setNearPlane(parseFloat(event.target.value));
            document.getElementById("nearVal").innerText = event.target.value;
        });
        farSlider.addEventListener("input", (event) => {
            this.camera.setFarPlane(parseFloat(event.target.value));
            document.getElementById("farVal").innerText = event.target.value;
        });
        // View angle
        const angleSlider = document.getElementById("angle");
        angleSlider.addEventListener("input", (event) => {
            this.camera.setViewAngle(parseFloat(event.target.value));
            document.getElementById("angleVal").innerText = event.target.value;
        });
    }

    updateCameraInfo() {
        document.getElementById("eyeX").value = this.camera.getEyePoint()[0];
        document.getElementById("eyeY").value = this.camera.getEyePoint()[1];
        document.getElementById("eyeZ").value = this.camera.getEyePoint()[2];
        document.getElementById("eyeXVal").innerText = eyeX.value;
        document.getElementById("eyeYVal").innerText = eyeY.value;
        document.getElementById("eyeZVal").innerText = eyeZ.value;
        document.getElementById("lookX").value = this.camera.getLookVector()[0];
        document.getElementById("lookY").value = this.camera.getLookVector()[1];
        document.getElementById("lookZ").value = this.camera.getLookVector()[2];
        document.getElementById("lookXVal").innerText = this.camera.getLookVector()[0];
        document.getElementById("lookYVal").innerText = this.camera.getLookVector()[1];
        document.getElementById("lookZVal").innerText = this.camera.getLookVector()[2];
        document.getElementById("near").value = this.camera.nearPlane;
        document.getElementById("far").value = this.camera.farPlane;
        document.getElementById("angle").value = this.camera.viewAngle;
        document.getElementById("nearVal").innerText = this.camera.nearPlane;
        document.getElementById("farVal").innerText = this.camera.farPlane;
        document.getElementById("angleVal").innerText = this.camera.viewAngle;
        document.getElementById("rotateU").value = this.camera.rotU;
        document.getElementById("rotateV").value = this.camera.rotV;
        document.getElementById("rotateW").value = this.camera.rotW;
        document.getElementById("rotateUVal").innerText = this.camera.rotU;
        document.getElementById("rotateVVal").innerText = this.camera.rotV;
        document.getElementById("rotateWVal").innerText = this.camera.rotW;
    }
}
