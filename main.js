import { WebGLRenderer } from './src/render.js';
import { PoseDetector } from './src/poseDetector.js';

// Usage
window.addEventListener('load', (async () => {
    renderer = new WebGLRenderer("glcanvas", "status", "xmlInput");
    try {
        await renderer.init();
    } catch (err) {
        console.error('Application initialization failed:', err);
    }

    // Initialize pose detector
    const poseDetector = new PoseDetector();
    try {
        await poseDetector.init();
    } catch (err) {
        console.error('Pose detector initialization failed:', err);
    }
}));