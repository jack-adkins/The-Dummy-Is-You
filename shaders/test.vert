#version 300 es

// In a fullscreen-triangle setup we don't need any 'in' attributes.
// gl_VertexID will take values 0, 1, 2, and the if/?: below converts it to vertex coordinates.
void main() {
    // Construct a full-screen triangle from gl_VertexID:
    //   Vertex 0 ⇒ (-1, -1)
    //   Vertex 1 ⇒ ( 3, -1)
    //   Vertex 2 ⇒ (-1,  3)
    //
    // This way, the three vertices will cover the entire screen in clip space:
    //   (-1,-1) ⇒ bottom-left, (3,-1) ⇒ bottom-right (outside canvas), (-1,3) ⇒ top-left (outside canvas)
    // The GPU will interpolate along the triangle edges to fill the entire screen area.
    vec2 pos = vec2(
        (gl_VertexID == 1) ? 3.0 : -1.0,
        (gl_VertexID == 2) ? 3.0 : -1.0
    );
    gl_Position = vec4(pos, 0.0, 1.0);
}