#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// ---------- Orbit camera state ----------
static float gYaw = glm::radians(35.0f);    // around Y axis
static float gPitch = glm::radians(25.0f);  // up/down
static float gDistance = 26.0f;             // zoom (radius)
static glm::vec3 gTarget(0.0f, 0.0f, 0.0f); // sun at origin

static bool gDragging = false;
static double gLastX = 0.0, gLastY = 0.0;

static float clampf(float x, float a, float b) { return std::max(a, std::min(b, x)); }

static void scroll_callback(GLFWwindow*, double /*xoffset*/, double yoffset) {
    // scroll up = zoom in
    gDistance *= (yoffset > 0.0) ? 0.90f : 1.10f;
    gDistance = clampf(gDistance, 3.0f, 200.0f);
}

static void mouse_button_callback(GLFWwindow* window, int button, int action, int /*mods*/) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            gDragging = true;
            glfwGetCursorPos(window, &gLastX, &gLastY);
        } else if (action == GLFW_RELEASE) {
            gDragging = false;
        }
    }
}

static void cursor_pos_callback(GLFWwindow* window, double xpos, double ypos) {
    if (!gDragging) return;

    double dx = xpos - gLastX;
    double dy = ypos - gLastY;
    gLastX = xpos;
    gLastY = ypos;

    const float sensitivity = 0.005f;
    gYaw   += (float)dx * sensitivity;
    gPitch += (float)dy * sensitivity;

    // clamp pitch so you don't flip
    gPitch = clampf(gPitch, glm::radians(-85.0f), glm::radians(85.0f));
}

static glm::vec3 orbit_camera_position() {
    // spherical coordinates -> cartesian
    float cp = std::cos(gPitch);
    glm::vec3 offset;
    offset.x = gDistance * cp * std::cos(gYaw);
    offset.y = gDistance * std::sin(gPitch);
    offset.z = gDistance * cp * std::sin(gYaw);
    return gTarget + offset;
}

// ----------------------- Sphere mesh (pos+normal) -----------------------
struct SphereMesh {
    std::vector<float> verts;        // interleaved: pos(3), normal(3)
    std::vector<unsigned int> idx;   // triangle indices
};

static SphereMesh make_uv_sphere(float radius, int stacks, int slices) {
    SphereMesh m;
    m.verts.reserve((stacks + 1) * (slices + 1) * 6);
    m.idx.reserve(stacks * slices * 6);

    for (int i = 0; i <= stacks; ++i) {
        float v = (float)i / (float)stacks;
        float phi = v * (float)M_PI;

        float y = std::cos(phi);
        float r = std::sin(phi);

        for (int j = 0; j <= slices; ++j) {
            float u = (float)j / (float)slices;
            float theta = u * 2.0f * (float)M_PI;

            float x = r * std::cos(theta);
            float z = r * std::sin(theta);

            // position
            m.verts.push_back(radius * x);
            m.verts.push_back(radius * y);
            m.verts.push_back(radius * z);

            // normal (unit)
            m.verts.push_back(x);
            m.verts.push_back(y);
            m.verts.push_back(z);
        }
    }

    int ring = slices + 1;
    for (int i = 0; i < stacks; ++i) {
        for (int j = 0; j < slices; ++j) {
            unsigned int a = i * ring + j;
            unsigned int b = (i + 1) * ring + j;
            unsigned int c = (i + 1) * ring + (j + 1);
            unsigned int d = i * ring + (j + 1);

            m.idx.push_back(a); m.idx.push_back(b); m.idx.push_back(d);
            m.idx.push_back(d); m.idx.push_back(b); m.idx.push_back(c);
        }
    }
    return m;
}

// ----------------------- Grid mesh (wireframe lines) --------------------
struct GridMesh {
    std::vector<float> verts;        // pos(3)
    std::vector<unsigned int> idx;   // line indices
};

static GridMesh make_grid(float halfSize, int divisions) {
    // divisions = number of cells along one axis
    // vertices = (divisions+1)^2
    GridMesh g;
    int n = divisions + 1;
    g.verts.reserve(n * n * 3);

    for (int iz = 0; iz < n; ++iz) {
        float tz = (float)iz / (float)divisions;             // 0..1
        float z = -halfSize + tz * (2.0f * halfSize);
        for (int ix = 0; ix < n; ++ix) {
            float tx = (float)ix / (float)divisions;
            float x = -halfSize + tx * (2.0f * halfSize);
            g.verts.push_back(x);
            g.verts.push_back(0.0f);
            g.verts.push_back(z);
        }
    }

    // line indices: connect horizontal and vertical neighbors
    // horizontal: (ix -> ix+1), vertical: (iz -> iz+1)
    g.idx.reserve((divisions * n + divisions * n) * 2);

    auto vid = [n](int ix, int iz) { return (unsigned int)(iz * n + ix); };

    for (int iz = 0; iz < n; ++iz) {
        for (int ix = 0; ix < n - 1; ++ix) {
            g.idx.push_back(vid(ix, iz));
            g.idx.push_back(vid(ix + 1, iz));
        }
    }
    for (int iz = 0; iz < n - 1; ++iz) {
        for (int ix = 0; ix < n; ++ix) {
            g.idx.push_back(vid(ix, iz));
            g.idx.push_back(vid(ix, iz + 1));
        }
    }
    return g;
}

// ----------------------- Shaders helpers --------------------------------
static void framebuffer_size_callback(GLFWwindow*, int w, int h) {
    glViewport(0, 0, w, h);
}

static GLuint compileShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);

    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(s, sizeof(log), nullptr, log);
        std::cerr << "Shader compile error:\n" << log << "\n";
        glDeleteShader(s);
        return 0;
    }
    return s;
}

static GLuint makeProgram(const char* vs, const char* fs) {
    GLuint v = compileShader(GL_VERTEX_SHADER, vs);
    if (!v) return 0;
    GLuint f = compileShader(GL_FRAGMENT_SHADER, fs);
    if (!f) { glDeleteShader(v); return 0; }

    GLuint p = glCreateProgram();
    glAttachShader(p, v);
    glAttachShader(p, f);
    glLinkProgram(p);

    glDeleteShader(v);
    glDeleteShader(f);

    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetProgramInfoLog(p, sizeof(log), nullptr, log);
        std::cerr << "Program link error:\n" << log << "\n";
        glDeleteProgram(p);
        return 0;
    }
    return p;
}

// ----------------------- N-body physics ---------------------------------
struct Body {
    float mass;
    float radius;      // visual radius
    glm::vec3 color;

    glm::vec3 pos;
    glm::vec3 vel;
    glm::vec3 acc;
};

static glm::vec3 accel_on(const std::vector<Body>& bodies, int i, float G, float softening) {
    glm::vec3 a(0.0f);
    const glm::vec3 pi = bodies[i].pos;

    for (int j = 0; j < (int)bodies.size(); ++j) {
        if (j == i) continue;
        glm::vec3 r = bodies[j].pos - pi;
        float r2 = glm::dot(r, r) + softening * softening;
        float invR = 1.0f / std::sqrt(r2);
        float invR3 = invR * invR * invR;
        a += G * bodies[j].mass * r * invR3;
    }
    return a;
}

static void compute_all_accels(std::vector<Body>& bodies, float G, float softening) {
    for (int i = 0; i < (int)bodies.size(); ++i) {
        bodies[i].acc = accel_on(bodies, i, G, softening);
    }
}

static void step_verlet(std::vector<Body>& bodies, float dt, float G, float softening) {
    // 1) x(t+dt) = x + v*dt + 0.5*a*dt^2
    for (auto& b : bodies) {
        b.pos += b.vel * dt + 0.5f * b.acc * (dt * dt);
    }

    // 2) a(t+dt)
    std::vector<glm::vec3> a_new(bodies.size());
    for (int i = 0; i < (int)bodies.size(); ++i) {
        a_new[i] = accel_on(bodies, i, G, softening);
    }

    // 3) v(t+dt) = v + 0.5*(a_old + a_new)*dt
    for (int i = 0; i < (int)bodies.size(); ++i) {
        bodies[i].vel += 0.5f * (bodies[i].acc + a_new[i]) * dt;
        bodies[i].acc = a_new[i];
    }
}

// ----------------------- Main -------------------------------------------
int main() {
    if (!glfwInit()) {
        std::cerr << "Failed to init GLFW\n";
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(1100, 700, "Solar System (Lit Spheres + Warped Grid)", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create window\n";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to init GLAD\n";
        glfwTerminate();
        return -1;
    }

    glfwSetScrollCallback(window, scroll_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_pos_callback);


    glEnable(GL_DEPTH_TEST);
    glClearColor(0.04f, 0.05f, 0.07f, 1.0f);

    // ----------- Shaders -----------
    const char* sphereVS = R"(
        #version 330 core
        layout (location=0) in vec3 aPos;
        layout (location=1) in vec3 aNormal;

        out vec3 FragPos;
        out vec3 Normal;

        uniform mat4 MVP;
        uniform mat4 Model;

        void main() {
            FragPos = vec3(Model * vec4(aPos, 1.0));
            Normal = mat3(transpose(inverse(Model))) * aNormal;
            gl_Position = MVP * vec4(aPos, 1.0);
        }
    )";

    const char* sphereFS = R"(
        #version 330 core
        out vec4 FragColor;

        in vec3 FragPos;
        in vec3 Normal;

        uniform vec3 lightPos;
        uniform vec3 viewPos;
        uniform vec3 color;

        void main() {
            vec3 norm = normalize(Normal);

            // ambient
            vec3 ambient = 0.18 * color;

            // diffuse
            vec3 lightDir = normalize(lightPos - FragPos);
            float diff = max(dot(norm, lightDir), 0.0);
            vec3 diffuse = diff * color;

            // specular
            vec3 viewDir = normalize(viewPos - FragPos);
            vec3 reflectDir = reflect(-lightDir, norm);
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), 48.0);
            vec3 specular = 0.6 * spec * vec3(1.0);

            vec3 result = ambient + diffuse + specular;
            FragColor = vec4(result, 1.0);
        }
    )";

    // Grid warp: wireframe lines (no lighting)
    // Warps Y by summed "potential wells"
    const int MAX_BODIES = 16;

    const char* gridVS = R"(
        #version 330 core
        layout (location=0) in vec3 aPos;

        out float vJupiterFrac;   // 0..1 how much is from Jupiter

        uniform mat4 MVP;

        uniform int bodyCount;
        uniform vec3 bodyPos[16];
        uniform float bodyMass[16];

        uniform float warpK;
        uniform float warpEps;

        // set this from C++ to 5
        uniform int jupiterIndex;

        void main() {
            vec3 p = aPos;

            float totalWell = 0.0;
            float jWell = 0.0;

            for (int i = 0; i < bodyCount; i++) {
                vec3 d = p - bodyPos[i];
                float r2 = dot(d,d) + warpEps * warpEps;
                float w = bodyMass[i] / sqrt(r2);
                totalWell += w;
                if (i == jupiterIndex) jWell = w;
            }

            // warp the grid
            p.y -= warpK * totalWell;

            // fraction of warp caused by Jupiter (avoid divide by 0)
            vJupiterFrac = (totalWell > 1e-6) ? (jWell / totalWell) : 0.0;

            gl_Position = MVP * vec4(p, 1.0);
        }

    )";

    const char* gridFS = R"(
        #version 330 core
        in float vJupiterFrac;
        out vec4 FragColor;

        uniform vec3 lineColor;    // your normal grid color

        void main() {
            // emphasize Jupiter influence: 0 -> normal color, 1 -> red
            float t = clamp(vJupiterFrac * 10.0, 0.0, 1.0); // *3 boosts visibility
            vec3 c = mix(lineColor, vec3(1.0, 0.15, 0.15), t);
            FragColor = vec4(c, 1.0);
        }

    )";

    GLuint sphereProg = makeProgram(sphereVS, sphereFS);
    GLuint gridProg   = makeProgram(gridVS, gridFS);
    if (!sphereProg || !gridProg) {
        glfwTerminate();
        return -1;
    }

    // ----------- Geometry -----------
    // Sphere (shared)
    SphereMesh sphere = make_uv_sphere(1.0f, 32, 64);

    GLuint sVAO=0, sVBO=0, sEBO=0;
    glGenVertexArrays(1, &sVAO);
    glGenBuffers(1, &sVBO);
    glGenBuffers(1, &sEBO);

    glBindVertexArray(sVAO);

    glBindBuffer(GL_ARRAY_BUFFER, sVBO);
    glBufferData(GL_ARRAY_BUFFER, sphere.verts.size()*sizeof(float), sphere.verts.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sphere.idx.size()*sizeof(unsigned int), sphere.idx.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6*sizeof(float), (void*)(3*sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);

    // Grid (wire)
    GridMesh grid = make_grid(/*halfSize=*/20.0f, /*divisions=*/220);

    GLuint gVAO=0, gVBO=0, gEBO=0;
    glGenVertexArrays(1, &gVAO);
    glGenBuffers(1, &gVBO);
    glGenBuffers(1, &gEBO);

    glBindVertexArray(gVAO);

    glBindBuffer(GL_ARRAY_BUFFER, gVBO);
    glBufferData(GL_ARRAY_BUFFER, grid.verts.size()*sizeof(float), grid.verts.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, grid.idx.size()*sizeof(unsigned int), grid.idx.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindVertexArray(0);

    // ----------- Bodies (scaled / toy solar system) -----------
    // Units are arbitrary and tuned for stability/looks.
    std::vector<Body> bodies;
    bodies.reserve(10);

    // Sun
    bodies.push_back({
        /*mass*/ 1200.0f,
        /*radius*/ 1.6f,
        /*color*/ {1.0f, 0.8f, 0.2f},
        /*pos*/   {0.0f, 0.0f, 0.0f},
        /*vel*/   {0.0f, 0.0f, 0.0f},
        /*acc*/   {0.0f, 0.0f, 0.0f}
    });

    // Helper: add near-circular orbit around sun in XZ plane
    auto addPlanet = [&](float dist, float mass, float radius, glm::vec3 color) {
        // circular v ~ sqrt(G*M/dist)
        // We'll set v along +Z when x=dist
        bodies.push_back({
            mass, radius, color,
            glm::vec3(dist, 0.0f, 0.0f),
            glm::vec3(0.0f, 0.0f, 0.0f),
            glm::vec3(0.0f)
        });
    };

    addPlanet(3.5f, 0.20f, 0.22f, {0.7f, 0.7f, 0.8f});  // Mercury-ish
    addPlanet(5.0f, 0.35f, 0.30f, {0.9f, 0.7f, 0.4f});  // Venus-ish
    addPlanet(7.0f, 0.40f, 0.32f, {0.2f, 0.6f, 1.0f});  // Earth-ish
    addPlanet(9.5f, 0.25f, 0.26f, {1.0f, 0.3f, 0.2f});  // Mars-ish
    addPlanet(13.0f, 2.50f, 0.75f, {0.9f, 0.7f, 0.5f}); // Jupiter-ish
    addPlanet(17.0f, 1.80f, 0.65f, {0.9f, 0.8f, 0.7f}); // Saturn-ish

    // Physics knobs
    float G = 2.0f;           // tuned gravitational constant
    float softening = 0.12f;  // avoids singularities
    float dt = 0.006f;        // simulation step (tuned)

    // Initialize circular velocities
    // v = sqrt(G * M_sun / r)
    for (int i = 1; i < (int)bodies.size(); ++i) {
        float r = glm::length(bodies[i].pos - bodies[0].pos);
        float v = std::sqrt(G * bodies[0].mass / std::max(r, 0.001f));
        bodies[i].vel = glm::vec3(0.0f, 0.0f, v);
    }
    compute_all_accels(bodies, G, softening);

    

    glm::vec3 lightPos(10.0f, 18.0f, 10.0f);

    // Uniform locations
    // Sphere
    GLint s_uMVP   = glGetUniformLocation(sphereProg, "MVP");
    GLint s_uModel = glGetUniformLocation(sphereProg, "Model");
    GLint s_uLight = glGetUniformLocation(sphereProg, "lightPos");
    GLint s_uView  = glGetUniformLocation(sphereProg, "viewPos");
    GLint s_uColor = glGetUniformLocation(sphereProg, "color");

    // Grid
    GLint g_uMVP     = glGetUniformLocation(gridProg, "MVP");
    GLint g_uCount   = glGetUniformLocation(gridProg, "bodyCount");
    GLint g_uPos     = glGetUniformLocation(gridProg, "bodyPos");
    GLint g_uMass    = glGetUniformLocation(gridProg, "bodyMass");
    GLint g_uK       = glGetUniformLocation(gridProg, "warpK");
    GLint g_uEps     = glGetUniformLocation(gridProg, "warpEps");
    GLint g_uLineCol = glGetUniformLocation(gridProg, "lineColor");
    GLint g_uJupiterIndex = glGetUniformLocation(gridProg, "jupiterIndex");

    // Warp visualization knobs (purely visual)
    float warpK = 0.15f;     // strength of dents
    float warpEps = 0.8f;    // smoothing for warp wells

    // ----------- Main loop -----------
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // Step physics (you can do multiple substeps if you want)
        step_verlet(bodies, dt, G, softening);

        int w, h;
        glfwGetFramebufferSize(window, &w, &h);
        float aspect = (h == 0) ? 1.0f : (float)w / (float)h;

        glm::vec3 cameraPos = orbit_camera_position();

        glm::mat4 View = glm::lookAt(
            cameraPos,
            gTarget,
            glm::vec3(0.0f, 1.0f, 0.0f)
        );

        glm::mat4 Projection = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 200.0f);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // ---- Draw grid first (wireframe lines) ----
        {
            glUseProgram(gridProg);

            glm::mat4 gridModel = glm::mat4(1.0f);
            glm::mat4 gridMVP = Projection * View * gridModel;
            glUniformMatrix4fv(g_uMVP, 1, GL_FALSE, glm::value_ptr(gridMVP));

            // Pack positions/masses into fixed-size arrays for uniforms
            int count = (int)std::min((size_t)MAX_BODIES, bodies.size());
            std::vector<glm::vec3> pos(count);
            std::vector<float> mass(count);

            for (int i = 0; i < count; ++i) {
                pos[i] = bodies[i].pos;
                mass[i] = bodies[i].mass;
            }

            glUniform1i(g_uCount, count);
            glUniform3fv(g_uPos, count, (const GLfloat*)pos.data());
            glUniform1fv(g_uMass, count, (const GLfloat*)mass.data());
            glUniform1f(g_uK, warpK);
            glUniform1f(g_uEps, warpEps);

            glUniform3f(g_uLineCol, 0.22f, 0.26f, 0.32f);
            glUniform1i(g_uJupiterIndex, 5);

            glBindVertexArray(gVAO);
            glDrawElements(GL_LINES, (GLsizei)grid.idx.size(), GL_UNSIGNED_INT, 0);
            glBindVertexArray(0);
        }

        // ---- Draw spheres (lit) ----
        {
            glUseProgram(sphereProg);

            glUniform3f(s_uLight, lightPos.x, lightPos.y, lightPos.z);
            glUniform3f(s_uView, cameraPos.x, cameraPos.y, cameraPos.z);

            glBindVertexArray(sVAO);

            for (const auto& b : bodies) {
                glm::mat4 Model = glm::translate(glm::mat4(1.0f), b.pos) *
                                  glm::scale(glm::mat4(1.0f), glm::vec3(b.radius));

                glm::mat4 MVP = Projection * View * Model;

                glUniformMatrix4fv(s_uMVP, 1, GL_FALSE, glm::value_ptr(MVP));
                glUniformMatrix4fv(s_uModel, 1, GL_FALSE, glm::value_ptr(Model));
                glUniform3f(s_uColor, b.color.r, b.color.g, b.color.b);

                glDrawElements(GL_TRIANGLES, (GLsizei)sphere.idx.size(), GL_UNSIGNED_INT, 0);
            }

            glBindVertexArray(0);
        }

        glfwSwapBuffers(window);
    }

    // Cleanup
    glDeleteProgram(sphereProg);
    glDeleteProgram(gridProg);

    glDeleteBuffers(1, &sEBO);
    glDeleteBuffers(1, &sVBO);
    glDeleteVertexArrays(1, &sVAO);

    glDeleteBuffers(1, &gEBO);
    glDeleteBuffers(1, &gVBO);
    glDeleteVertexArrays(1, &gVAO);

    glfwTerminate();
    return 0;
}



