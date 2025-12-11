import { Camera } from './camera.js';
import { Controls } from './controls.js';
import { XMLSceneParser } from './XMLSceneParser.js';
import { SceneFlattener } from './SceneFlattener.js';
import { ShaderProgram } from './shaderProgram.js';
import { loadPPMFromText } from './ppm.js';
import { PoseReceiver } from './poseExporter.js';
import { PrimitiveType } from './SceneDataStructures.js';

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

        // Pose tracking
        this.poseReceiver = null;
        // Skeleton structure: 1 head sphere + 1 torso cylinder + 12 joint spheres + 8 bone cylinders = 22 objects
        this.poseObjectCount = 22; // Will be set to skeletonStructure.length below
        this.baseObjectCount = 0; // Objects from XML scene
        this.poseObjectStartIndex = 0; // Index where pose objects start in texture
        this.lastPoseData = null;
        this.poseDataArray = null; // Float32Array for pose objects data
        
        // Skeleton structure definition
        this.skeletonStructure = [
            { type: 'head', primitive: PrimitiveType.SHAPE_SPHERE, landmark: 0 }, // nose
            { type: 'torso', primitive: PrimitiveType.SHAPE_CYLINDER, from: [23, 24], to: [11, 12] }, // hip center to shoulder center
            { type: 'joint', primitive: PrimitiveType.SHAPE_SPHERE, landmark: 11 }, // left shoulder
            { type: 'joint', primitive: PrimitiveType.SHAPE_SPHERE, landmark: 13 }, // left elbow
            { type: 'joint', primitive: PrimitiveType.SHAPE_SPHERE, landmark: 15 }, // left wrist
            { type: 'joint', primitive: PrimitiveType.SHAPE_SPHERE, landmark: 12 }, // right shoulder
            { type: 'joint', primitive: PrimitiveType.SHAPE_SPHERE, landmark: 14 }, // right elbow
            { type: 'joint', primitive: PrimitiveType.SHAPE_SPHERE, landmark: 16 }, // right wrist
            { type: 'joint', primitive: PrimitiveType.SHAPE_SPHERE, landmark: 23 }, // left hip
            { type: 'joint', primitive: PrimitiveType.SHAPE_SPHERE, landmark: 25 }, // left knee
            { type: 'joint', primitive: PrimitiveType.SHAPE_SPHERE, landmark: 27 }, // left ankle
            { type: 'joint', primitive: PrimitiveType.SHAPE_SPHERE, landmark: 24 }, // right hip
            { type: 'joint', primitive: PrimitiveType.SHAPE_SPHERE, landmark: 26 }, // right knee
            { type: 'joint', primitive: PrimitiveType.SHAPE_SPHERE, landmark: 28 }, // right ankle
            // Bones (cylinders)
            { type: 'bone', primitive: PrimitiveType.SHAPE_CYLINDER, from: 11, to: 13 }, // left upper arm
            { type: 'bone', primitive: PrimitiveType.SHAPE_CYLINDER, from: 13, to: 15 }, // left lower arm
            { type: 'bone', primitive: PrimitiveType.SHAPE_CYLINDER, from: 12, to: 14 }, // right upper arm
            { type: 'bone', primitive: PrimitiveType.SHAPE_CYLINDER, from: 14, to: 16 }, // right lower arm
            { type: 'bone', primitive: PrimitiveType.SHAPE_CYLINDER, from: 23, to: 25 }, // left upper leg
            { type: 'bone', primitive: PrimitiveType.SHAPE_CYLINDER, from: 25, to: 27 }, // left lower leg
            { type: 'bone', primitive: PrimitiveType.SHAPE_CYLINDER, from: 24, to: 26 }, // right upper leg
            { type: 'bone', primitive: PrimitiveType.SHAPE_CYLINDER, from: 26, to: 28 }  // right lower leg
        ];
        
        // Update count to match structure
        this.poseObjectCount = this.skeletonStructure.length;

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
        this.setupPoseReceiver();       // setup pose data receiver
        this.startRenderLoop();         // start the render loop
    }

    setupPoseReceiver() {
        this.poseReceiver = new PoseReceiver((poseData) => {
            this.lastPoseData = poseData;
        });
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
            this.programs["raytrace"].reload(vsText, fsText);
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

                // Store base scene info
                this.baseObjectCount = objectCount;
                
                // pass the flattened data to the shader program (includes pose objects)
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
        
        // Total objects = base scene objects + pose objects
        const totalObjectCount = objectCount + this.poseObjectCount;
        const texWidth = floatsPerRow / 4;
        const texHeight = totalObjectCount;

        // Pad the data length to match texWidth * texHeight * 4
        const totalFloats = floatsPerRow * totalObjectCount;
        const dataArray = new Float32Array(totalFloats);

        // Copy base scene objects
        for (let i = 0; i < objectCount; ++i) {
            const srcOffset = i * floatsPerObject;
            const dstOffset = i * floatsPerRow;
            dataArray.set(
                flatArray.subarray(srcOffset, srcOffset + floatsPerObject),
                dstOffset
            );
        }

        // Initialize pose objects (spheres at origin with default material)
        this.poseObjectStartIndex = objectCount;
        this.initializePoseObjects(dataArray, objectCount, floatsPerObject, floatsPerRow);

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
        this.objectCount = totalObjectCount;
        this.floatsPerObject = floatsPerObject;
        this.floatsPerRow = floatsPerRow;
        this.texWidth = texWidth;
        this.texHeight = texHeight;
        
        // Store reference to pose data array for updates
        this.poseDataArray = new Float32Array(floatsPerRow * this.poseObjectCount);
    }

    initializePoseObjects(dataArray, startIndex, floatsPerObject, floatsPerRow) {
        // Initialize each pose object based on skeleton structure
        for (let i = 0; i < this.poseObjectCount; ++i) {
            const objIndex = startIndex + i;
            const offset = objIndex * floatsPerRow;
            const part = this.skeletonStructure[i];
            
            // Type based on skeleton part
            dataArray[offset] = part.primitive;
            
            // World matrix: identity (will be updated with transformations)
            for (let j = 0; j < 16; j++) {
                dataArray[offset + 1 + j] = (j % 5 === 0) ? 1.0 : 0.0; // Identity matrix
            }
            
            // Material properties (18 floats)
            const matOffset = offset + 17;
            // Ambient (r, g, b)
            dataArray[matOffset] = 0.2;
            dataArray[matOffset + 1] = 0.2;
            dataArray[matOffset + 2] = 0.2;
            
            // Diffuse color based on part type
            let rgb;
            if (part.type === 'head') {
                rgb = [1.0, 0.8, 0.6]; // Skin tone
            } else if (part.type === 'torso') {
                rgb = [0.2, 0.4, 0.8]; // Blue
            } else if (part.type === 'joint') {
                rgb = [0.9, 0.9, 0.9]; // White joints
            } else { // bone
                rgb = [0.6, 0.6, 0.6]; // Gray bones
            }
            
            dataArray[matOffset + 3] = rgb[0];
            dataArray[matOffset + 4] = rgb[1];
            dataArray[matOffset + 5] = rgb[2];
            // Specular (r, g, b)
            dataArray[matOffset + 6] = 0.5;
            dataArray[matOffset + 7] = 0.5;
            dataArray[matOffset + 8] = 0.5;
            // Shininess
            dataArray[matOffset + 9] = 30.0;
            // IOR
            dataArray[matOffset + 10] = 1.0;
            // Texture map (not used)
            dataArray[matOffset + 11] = 0.0;
            dataArray[matOffset + 12] = 1.0; // repeatU
            dataArray[matOffset + 13] = 1.0; // repeatV
            dataArray[matOffset + 14] = 0.0; // textureIndex
            // Reflective (r, g, b)
            dataArray[matOffset + 15] = 0.0;
            dataArray[matOffset + 16] = 0.0;
            dataArray[matOffset + 17] = 0.0;
        }
    }

    hslToRgb(h, s, l) {
        let r, g, b;
        if (s === 0) {
            r = g = b = l;
        } else {
            const hue2rgb = (p, q, t) => {
                if (t < 0) t += 1;
                if (t > 1) t -= 1;
                if (t < 1/6) return p + (q - p) * 6 * t;
                if (t < 1/2) return q;
                if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
                return p;
            };
            const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
            const p = 2 * l - q;
            r = hue2rgb(p, q, h + 1/3);
            g = hue2rgb(p, q, h);
            b = hue2rgb(p, q, h - 1/3);
        }
        return [r, g, b];
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

    updatePoseObjects(poseData) {
        if (!poseData || !poseData.poses || poseData.poses.length === 0) {
            return;
        }

        if (!this.sceneTexture || !this.poseDataArray) {
            return;
        }

        const pose = poseData.poses[0];
        const landmarks = pose.worldLandmarks && pose.worldLandmarks.length > 0
            ? pose.worldLandmarks
            : pose.landmarks;

        if (landmarks.length < 33) {
            return;
        }

        const gl = this.gl;
        const floatsPerRow = this.floatsPerRow;

        // Helper function to convert landmark to 3D position
        const landmarkToPos = (landmark) => {
            if (!landmark) return [1000, 1000, 1000]; // Hide if no landmark
           
            if (pose.worldLandmarks && pose.worldLandmarks.length > 0) {
                // Use world landmarks (3D space in meters)
                return [
                    -landmark.x * 2.0,           // Scale and mirror X
                    -landmark.y * 2.0 + 1.5,    // Scale, mirror Y, and offset up
                    -landmark.z * 2.0           // Scale and mirror Z
                ];
            } else {
                // Use normalized screen coordinates (0-1)
                return [
                    (landmark.x - 0.5) * 4.0,    // Scale to scene size
                    -(landmark.y - 0.5) * 4.0 + 1.5, // Invert Y and offset
                    -landmark.z * 4.0            // Scale depth
                ];
            }
        };

        // Helper function to get landmark by index
        const getLandmark = (idx) => {
            if (Array.isArray(idx)) {
                // Average of multiple landmarks
                let sumX = 0, sumY = 0, sumZ = 0, count = 0;
                for (const i of idx) {
                    if (i < landmarks.length && landmarks[i]) {
                        const pos = landmarkToPos(landmarks[i]);
                        sumX += pos[0];
                        sumY += pos[1];
                        sumZ += pos[2];
                        count++;
                    }
                }
                return count > 0 ? [sumX / count, sumY / count, sumZ / count] : [1000, 1000, 1000];
            } else {
                return idx < landmarks.length ? landmarkToPos(landmarks[idx]) : [1000, 1000, 1000];
            }
        };

        // Helper function to create transformation matrix for cylinder
        // Cylinder is unit size (radius 1, height 1) along Y axis, centered at origin
        const createCylinderMatrix = (fromPos, toPos, radius = 0.05, heightScale = 1.5) => {
            const dir = [
                toPos[0] - fromPos[0],
                toPos[1] - fromPos[1],
                toPos[2] - fromPos[2]
            ];
            const length = Math.sqrt(dir[0]*dir[0] + dir[1]*dir[1] + dir[2]*dir[2]);
           
            if (length < 0.001) {
                // Too short, hide it
                return new Float32Array([
                    0.001, 0, 0, 1000,
                    0, 0.001, 0, 1000,
                    0, 0, 0.001, 1000,
                    0, 0, 0, 1
                ]);
            }
           
            // Normalize direction (this is the new Y axis)
            const yAxis = [dir[0] / length, dir[1] / length, dir[2] / length];
           
            // Find a perpendicular vector for X axis
            let xAxis;
            if (Math.abs(yAxis[0]) < 0.9) {
                // Use [1,0,0] as reference
                const ref = [1, 0, 0];
                xAxis = [
                    yAxis[1] * ref[2] - yAxis[2] * ref[1],
                    yAxis[2] * ref[0] - yAxis[0] * ref[2],
                    yAxis[0] * ref[1] - yAxis[1] * ref[0]
                ];
            } else {
                // Use [0,1,0] as reference
                const ref = [0, 1, 0];
                xAxis = [
                    yAxis[1] * ref[2] - yAxis[2] * ref[1],
                    yAxis[2] * ref[0] - yAxis[0] * ref[2],
                    yAxis[0] * ref[1] - yAxis[1] * ref[0]
                ];
            }
           
            // Normalize X axis
            const xLen = Math.sqrt(xAxis[0]*xAxis[0] + xAxis[1]*xAxis[1] + xAxis[2]*xAxis[2]);
            if (xLen > 0.001) {
                xAxis[0] /= xLen;
                xAxis[1] /= xLen;
                xAxis[2] /= xLen;
            } else {
                xAxis = [1, 0, 0];
            }
           
            // Calculate Z axis (cross product)
            const zAxis = [
                xAxis[1] * yAxis[2] - xAxis[2] * yAxis[1],
                xAxis[2] * yAxis[0] - xAxis[0] * yAxis[2],
                xAxis[0] * yAxis[1] - xAxis[1] * yAxis[0]
            ];
           
            // Center position
            const center = [
                (fromPos[0] + toPos[0]) / 2,
                (fromPos[1] + toPos[1]) / 2,
                (fromPos[2] + toPos[2]) / 2
            ];
           
            // Transformation: T * R * S
            // Scale: radius in X/Z, length/2 in Y (cylinder height is 1, so scale by length/2)
            // Rotation: align Y axis with bone direction
            // Translation: to center
            // heightScale multiplies the length to make cylinder taller
            // Row-major format
            return new Float32Array([
                xAxis[0] * radius, yAxis[0] * (length / 2 * heightScale), zAxis[0] * radius, center[0],
                xAxis[1] * radius, yAxis[1] * (length / 2 * heightScale), zAxis[1] * radius, center[1],
                xAxis[2] * radius, yAxis[2] * (length / 2 * heightScale), zAxis[2] * radius, center[2],
                0, 0, 0, 1
            ]);
        };

        // Update each skeleton part
        for (let i = 0; i < this.poseObjectCount; i++) {
            const offset = i * floatsPerRow;
            const part = this.skeletonStructure[i];
            let matrix;
           
            if (part.type === 'head' || part.type === 'joint') {
                // Sphere at landmark position with scale
                const pos = getLandmark(part.landmark);
                const baseJointScale = 0.24; // Base size for joints
                const scale = part.type === 'head' ? baseJointScale * 3.0 : baseJointScale; // Head is 4x joints
                // Scale and translation matrix (row-major)
                matrix = new Float32Array([
                    scale, 0, 0, pos[0],
                    0, scale, 0, pos[1],
                    0, 0, scale, pos[2],
                    0, 0, 0, 1
                ]);
            } else if (part.type === 'torso') {
                // Cylinder from hip center to shoulder center
                const hipCenter = getLandmark(part.from);
                const shoulderCenter = getLandmark(part.to);
                const baseJointScale = 0.16; // Base size for joints
                const torsoRadius = baseJointScale * 4.0; // Torso is 4x joints
                // Make torso taller by extending the length
                matrix = createCylinderMatrix(hipCenter, shoulderCenter, torsoRadius, 2.0);
            } else if (part.type === 'bone') {
                // Cylinder between two landmarks
                const fromPos = getLandmark(part.from);
                const toPos = getLandmark(part.to);
                // Make arms and legs 1.5x larger (radius)
                const boneRadius = 0.05 * 2.0; // 1.5x larger
                matrix = createCylinderMatrix(fromPos, toPos, boneRadius);
            } else {
                // Default identity
                matrix = new Float32Array([
                    1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1
                ]);
            }
           
            // Type
            this.poseDataArray[offset] = part.primitive;
           
            // World matrix (16 floats, row-major)
            for (let j = 0; j < 16; j++) {
                this.poseDataArray[offset + 1 + j] = matrix[j];
            }
           
            // Material properties (keep from initialization)
            const matOffset = offset + 17;
            // Ambient
            this.poseDataArray[matOffset] = 0.2;
            this.poseDataArray[matOffset + 1] = 0.2;
            this.poseDataArray[matOffset + 2] = 0.2;
           
            // Diffuse color based on part type
            let rgb;
            if (part.type === 'head') {
                rgb = [1.0, 1.0, 0.0]; // yellow
            } else if (part.type === 'torso') {
                rgb = [0.6, 0.6, 0.6]; // grey
            } else if (part.type === 'joint') {
                rgb = [1.0, 0.0, 0.0]; // red
            } else { 
                rgb = [0.0, 1.0, 1.0]; // cyan
            }
           
            this.poseDataArray[matOffset + 3] = rgb[0];
            this.poseDataArray[matOffset + 4] = rgb[1];
            this.poseDataArray[matOffset + 5] = rgb[2];
            // Specular
            this.poseDataArray[matOffset + 6] = 0.5;
            this.poseDataArray[matOffset + 7] = 0.5;
            this.poseDataArray[matOffset + 8] = 0.5;
            // Shininess
            this.poseDataArray[matOffset + 9] = 30.0;
            // IOR
            this.poseDataArray[matOffset + 10] = 1.0;
            // Texture map
            this.poseDataArray[matOffset + 11] = 0.0;
            this.poseDataArray[matOffset + 12] = 1.0;
            this.poseDataArray[matOffset + 13] = 1.0;
            this.poseDataArray[matOffset + 14] = 0.0;
            // Reflective
            this.poseDataArray[matOffset + 15] = 0.0;
            this.poseDataArray[matOffset + 16] = 0.0;
            this.poseDataArray[matOffset + 17] = 0.0;
        }

        // Update texture with new pose data using texSubImage2D
        gl.bindTexture(gl.TEXTURE_2D, this.sceneTexture);
        gl.texSubImage2D(
            gl.TEXTURE_2D,
            0,
            0, // x offset
            this.poseObjectStartIndex, // y offset (row where pose objects start)
            this.texWidth, // width
            this.poseObjectCount, // height (number of pose objects)
            gl.RGBA,
            gl.FLOAT,
            this.poseDataArray
        );
        gl.bindTexture(gl.TEXTURE_2D, null);
    }

    renderFrame() {
        const gl = this.gl;
        this.resizeCanvasToDisplaySize();
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        if (!this.sceneReady) {
            this.statusElem.textContent = 'Waiting for scene to load...';
            return;
        }
        
        // Update pose objects if we have new pose data
        if (this.lastPoseData) {
            this.updatePoseObjects(this.lastPoseData);
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