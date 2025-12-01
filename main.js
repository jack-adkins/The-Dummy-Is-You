import { WebGLRenderer } from './src/render.js';

// Usage
window.addEventListener('load', (async () => {
    renderer = new WebGLRenderer("glcanvas", "status", "xmlInput");
    try {
        await renderer.init();
    } catch (err) {
        console.error('Application initialization failed:', err);
    }
}));