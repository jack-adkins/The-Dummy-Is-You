import { Camera } from './camera.js';
import { Controls } from './controls.js';
import { XMLSceneParser } from './XMLSceneParser.js';
import { SceneFlattener } from './SceneFlattener.js';
import { ShaderProgram } from './shaderProgram.js';
import { loadPPMFromText } from './ppm.js';

export class WebGLRenderer {
    constructor(canvasId, statusId, xmlInputId) {
        this.canvas = document.getElementById(canvasId);     // make a canvas 
        this.statusElem = document.getElementById(statusId); // tells user status, eg file loaded successfully
        this.xmlInput = document.getElementById(xmlInputId); // user uploads XML scene file
        this.gl = null;     // webgl context
        this.programs = {   // shader program manager 
            rayTrace: null,
        };
        this.camera = new Camera(); // our own camera class; handles transformations
        this.controls = new Controls(this);

        this.sceneParser = new XMLSceneParser(); // parses XML scene files
        this.sceneFlattener = null; // flattens the scene into a Float32Array for rendering
        this.sceneTexture = null;
        this.textures = []          // store WebGLTexture handles 
        this.sceneReady = false;    // whether the scene is ready to be rendered
        this.fullScreenVAO = null;  // vertex array object for full-screen quad rendering

        this.maxDepth = 2;         // maximum recursion depth

        this.uniformsLogged = false; // debug

        // below will be set by the scene parser
        this.objectCount = 0;
        this.floatsPerObject = 0;
        this.floatsPerRow = 0;
        this.texWidth = 0;
        this.texHeight = 0;

        this.init();
    }

    async init() {
        if (!this.initGL()) return; // initialize WebGL context, set background color, etc.
        try {
            await this.setupShaders();
        } catch (e) {
            console.error('Shader initialization failed:', e);
            this.statusElem.textContent = 'Shader initialization failed: ' + e.message;
            return;
        }
        this.setupFullScreenTriangle(); // setup a full-screen triangle for rendering
        this.setupEventHandlers();      // setup event handlers for user input
        this.startRenderLoop();         // start the render loop
    }

    initGL() {
        this.gl = this.canvas.getContext('webgl2');
        if (!this.gl) {
            this.statusElem.textContent = "Error: WebGL2 not supported in this browser.";
            return false;
        }
        this.gl.clearColor(0.1, 0.1, 0.1, 1.0); // background color is dark grey by default 
        this.gl.enable(this.gl.DEPTH_TEST);     // should we leave this on? 
        return true;
    }

    async setupShaders() {
        const gl = this.gl;
        const vsText = await fetch('./shaders/test.vert').then((r) => r.text());
        const fsText = await fetch('./shaders/test.frag').then((r) => r.text());
        this.programs.rayTrace = new ShaderProgram(gl, vsText, fsText);
    }

    async reloadShaders(name) {
        try {
            const vsText = await fetch(`./shaders/${name}.vert`).then((r) => r.text());
            const fsText = await fetch(`./shaders/${name}.frag`).then((r) => r.text());
            this.programs[name].reload(vsText, fsText);
            this.statusElem.textContent = "Shaders reloaded successfully!";
        } catch (error) {
            console.error("Shader reload failed:", error);
            this.statusElem.textContent =
                "Shader reload failed: " + error.message;
        }
    }

    setupEventHandlers() {
        if (this.xmlInput) {
            this.xmlInput.addEventListener('change', (evt) => this.handleXMLFileInput(evt));
        }
        else {
            console.error('XML input element not found. Please check the HTML.');
            this.statusElem.textContent = 'Error: XML input element not found.';
        }
    }

    async handleXMLFileInput(evt) {
        const fileList = evt.target.files;
        if (!fileList || fileList.length === 0) {
            this.statusElem.textContent = 'No file selected';
            return;
        }
        const xmlFile = fileList[0];
        if (!xmlFile.name.endsWith('.xml')) {
            this.statusElem.textContent = 'Please select a valid XML file';
            return;
        }

        const reader = new FileReader();
        reader.onload = async (loadEvt) => {
            try {
                const xmlText = loadEvt.target.result;
                this.sceneParser = new XMLSceneParser(); // reset parser for new file
                const parseOk = await this.sceneParser.parseFromString(xmlText);
                if (!parseOk) throw new Error('XMLSceneParser error: parsing failed');
                const rootNode = this.sceneParser.getRootNode();

                // flatten the scene
                this.sceneFlattener = new SceneFlattener(rootNode);
                this.sceneFlattener.flatten();
                const flatArray = this.sceneFlattener.getFloat32Array();
                const objectCount = this.sceneFlattener.getObjectCount();
                const floatsPerObject = this.sceneFlattener.floatsPerObject;
                console.log(`Flatten Array: ${flatArray}`);

                // pass the flattened data to the shader program
                this.createSceneDataTexture(flatArray, objectCount, floatsPerObject);

                // new in a4: load all PPMs referenced in the scene
                const gl = this.gl;
                const maps = this.sceneFlattener.getTextureMaps();
                this.textures = await Promise.all(maps.map(async (map) => {
                    const text = await fetch(map.filename).then(r => r.text());
                    const { tex } = loadPPMFromText(gl, text);
                    return tex;
                }));

                // set up the camera
                this.camera.reset();
                const cameraData = this.sceneParser.getCameraData();
                if (cameraData.isDir) {
                    this.camera.orientLookVec(cameraData.pos, cameraData.look, cameraData.up);
                    this.controls.updateCameraInfo();
                }
                else {
                    this.camera.orientLookAt(cameraData.pos, cameraData.lookAt, cameraData.up);
                    this.controls.updateCameraInfo();
                }

                // // new in a4: load scene textures from parser 
                // const texImages = this.sceneParser.getTextures();
                // this.textures = [];
                // texImages.forEach((img, ti))

                this.sceneReady = true;
                this.statusElem.textContent = `Scene loaded successfully: ${objectCount} objects, ${floatsPerObject} floats per object`;
            } catch (e) {
                console.error('Error loading or flattening scene:', e);
                this.statusElem.textContent = 'Error loading scene: ' + e.message;
            }
        };
        reader.onerror = (err) => {
            console.error('File reading error:', err);
            this.statusElem.textContent = 'Error reading file: ' + err.message;
        };
        reader.readAsText(xmlFile);
    }

    createSceneDataTexture(flatArray, objectCount, floatsPerObject) {
        const gl = this.gl;
        // Make sure each row is a multiple of 4 floats (for RGBA32F)
        const floatsPerRow = Math.ceil(floatsPerObject / 4) * 4;
        const texWidth = floatsPerRow / 4;
        const texHeight = objectCount;

        // Pad the data length to match texWidth * texHeight * 4
        const totalFloats = floatsPerRow * objectCount;
        const dataArray = new Float32Array(totalFloats);

        for (let i = 0; i < objectCount; ++i) {
            const srcOffset = i * floatsPerObject;
            const dstOffset = i * floatsPerRow;
            dataArray.set(
                flatArray.subarray(srcOffset, srcOffset + floatsPerObject),
                dstOffset
            );
        }

        // Create the texture
        const tex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.texImage2D(
            gl.TEXTURE_2D,
            0,
            gl.RGBA32F,
            texWidth,
            texHeight,
            0,
            gl.RGBA,
            gl.FLOAT,
            dataArray
        );
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.bindTexture(gl.TEXTURE_2D, null);

        // Store the texture and metadata in the renderer
        this.sceneTexture = tex;
        this.objectCount = objectCount;
        this.floatsPerObject = floatsPerObject;
        this.floatsPerRow = floatsPerRow;
        this.texWidth = texWidth;
        this.texHeight = texHeight;
    }

    setupFullScreenTriangle() {
        const gl = this.gl;
        const vao = gl.createVertexArray();
        gl.bindVertexArray(vao);
        // No need to bind any VBO, the vertex shader will generate vertices using gl_VertexID
        gl.bindVertexArray(null);
        this.fullScreenVAO = vao;
    }

    // Debugging helper: log all uniform values after first binding
    _logAllUniforms() {
        const gl = this.gl;
        const program = this.programs.rayTrace.program;

        const read = (name) => {
            const loc = this.programs.rayTrace.getUniformLocation(name);
            if (loc === null) {
                console.warn(`Uniform "${name}" does not exist or was not compiled into the shader`);
                return null;
            }
            return gl.getUniform(program, loc);
        };

        console.group('>>> WebGLRenderer Uniform Values After First Binding <<<');
        console.log('uResolution       =', read('uResolution'));
        console.log('uCameraPos        =', read('uCameraPos'));
        console.log('uInvProjView      =', read('uInvProjView'));
        console.log('uCamWorldMatrix   =', read('uCamWorldMatrix'));
        console.log('uGlobalKa         =', read('uGlobalKa'));
        console.log('uGlobalKd         =', read('uGlobalKd'));
        console.log('uGlobalKs         =', read('uGlobalKs'));
        console.log('uGlobalKt         =', read('uGlobalKt'));

        // Scene texture
        console.log('uSceneBuffer      =', read('uSceneBuffer'));

        // Scene metadata
        console.log('uObjectCount      =', read('uObjectCount'));
        console.log('uFloatsPerRow     =', read('uFloatsPerRow'));
        console.log('uSceneTexWidth    =', read('uSceneTexWidth'));
        console.log('uSceneTexHeight   =', read('uSceneTexHeight'));

        // lights
        const numLights = read('uNumLights');
        console.log('uNumLights        =', numLights);
        for (let i = 0; i < (numLights || 0) && i < 16; i++) {
            console.group(`--- Light[${i}] ---`);
            console.log(`uLightType[${i}]    =`, read(`uLightType[${i}]`));
            console.log(`uLightColor[${i}]   =`, read(`uLightColor[${i}]`));
            console.log(`uLightPos[${i}]     =`, read(`uLightPos[${i}]`));
            console.log(`uLightDir[${i}]     =`, read(`uLightDir[${i}]`));
            console.log(`uLightRadius[${i}]  =`, read(`uLightRadius[${i}]`));
            console.log(`uLightPenumbra[${i}] =`, read(`uLightPenumbra[${i}]`));
            console.log(`uLightAngle[${i}]    =`, read(`uLightAngle[${i}]`));
            console.log(`uLightWidth[${i}]    =`, read(`uLightWidth[${i}]`));
            console.log(`uLightHeight[${i}]   =`, read(`uLightHeight[${i}]`));
            console.groupEnd();
        }
        console.groupEnd();
    }

    startRenderLoop() {
        const loop = (t) => {
            this.renderFrame(t);
            requestAnimationFrame(loop);
        };
        requestAnimationFrame(loop);
    }

    resizeCanvasToDisplaySize() {
        const dpr = window.devicePixelRatio || 1;
        const displayWidth = Math.round(this.canvas.clientWidth * dpr);
        const displayHeight = Math.round(this.canvas.clientHeight * dpr);
        if (this.canvas.width !== displayWidth || this.canvas.height !== displayHeight) {
            this.canvas.width = displayWidth;
            this.canvas.height = displayHeight;
            this.gl.viewport(0, 0, displayWidth, displayHeight);
        }
    }

    resetScene() {
        this.camera.reset();
        const cam = this.sceneParser.getCameraData();
        if (cam.isDir) {
            this.camera.orientLookVec(cam.pos, cam.look, cam.up);
        } else {
            this.camera.orientLookAt(cam.pos, cam.lookAt, cam.up);
        }
        this.controls.updateCameraInfo();
    }

    renderFrame() {
        const gl = this.gl;
        this.resizeCanvasToDisplaySize();
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        if (!this.sceneReady) {
            this.statusElem.textContent = 'Waiting for scene to load...';
            return;
        }
        this.statusElem.textContent = 'Rendering...';

        // Use the ray tracing shader program
        this.gl.useProgram(this.programs.rayTrace.program);

        const width = this.canvas.width;
        const height = this.canvas.height;
        const uResolutionLoc = this.programs.rayTrace.getUniformLocation('uResolution');
        gl.uniform2f(uResolutionLoc, width, height);

        // Camera
        this.camera.setScreenSize(this.canvas.width, this.canvas.height);
        const MV = mat4.create();
        mat4.multiply(MV, this.camera.getScaleMatrix(), this.camera.getModelViewMatrix());
        const invMV = mat4.create();
        mat4.invert(invMV, MV);
        const camPos = this.camera.getEyePoint();

        this.programs.rayTrace.setVector3('uCameraPos', camPos);
        this.programs.rayTrace.setMatrix4('uCamWorldMatrix', invMV);

        // Global coefficients
        const globalData = this.sceneParser.getGlobalData();
        this.programs.rayTrace.setFloat('uGlobalKa', globalData.ka);
        this.programs.rayTrace.setFloat('uGlobalKd', globalData.kd);
        this.programs.rayTrace.setFloat('uGlobalKs', globalData.ks);
        this.programs.rayTrace.setFloat('uGlobalKt', globalData.kt);

        // Pass maximum recursion depth.
        gl.uniform1i(
            this.programs.rayTrace.getUniformLocation("uMaxDepth"),
            this.maxDepth
        );

        // Bind each ppm texture into units 1 2 3 etc
        this.textures.forEach((tex, i) => {
            gl.activeTexture(gl.TEXTURE1 + i);
            gl.bindTexture(gl.TEXTURE_2D, tex);
            this.programs.rayTrace.setInteger(`uTextures[${i}]`, 1 + i);
        });

        // Scene Texture
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.sceneTexture);
        gl.uniform1i(this.programs.rayTrace.getUniformLocation('uSceneBuffer'), 0);

        // Scene metadata
        this.programs.rayTrace.setInteger('uObjectCount', this.objectCount);
        this.programs.rayTrace.setInteger('uFloatsPerRow', this.floatsPerRow);
        this.programs.rayTrace.setInteger('uSceneTexWidth', this.texWidth);
        this.programs.rayTrace.setInteger('uSceneTexHeight', this.texHeight);

        // Lights
        const lights = this.sceneParser.getLights();
        const numLights = lights.length;
        this.programs.rayTrace.setInteger('uNumLights', numLights);
        for (let i = 0; i < numLights && i < 16; i++) {
            const L = lights[i];
            this.programs.rayTrace.setInteger(`uLightType[${i}]`, L.type);
            this.programs.rayTrace.setVector3(`uLightColor[${i}]`, [L.color.r, L.color.g, L.color.b]);
            this.programs.rayTrace.setVector3(`uLightPos[${i}]`, L.pos);
            this.programs.rayTrace.setVector3(`uLightDir[${i}]`, L.dir);
            this.programs.rayTrace.setFloat(`uLightRadius[${i}]`, L.radius);
            this.programs.rayTrace.setFloat(`uLightPenumbra[${i}]`, L.penumbra);
            this.programs.rayTrace.setFloat(`uLightAngle[${i}]`, L.angle);
            this.programs.rayTrace.setFloat(`uLightWidth[${i}]`, L.width);
            this.programs.rayTrace.setFloat(`uLightHeight[${i}]`, L.height);
        }

        if (!this.uniformsLogged) {
            this._logAllUniforms();
            this.uniformsLogged = true;
        }

        // Draw the full-screen triangle
        gl.bindVertexArray(this.fullScreenVAO);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        gl.bindVertexArray(null);
        gl.bindTexture(gl.TEXTURE_2D, null);
    }
}