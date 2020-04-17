//=============================================================================================
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Conforti Christian
// Neptun : F8R430
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================

#include "framework.h"

#pragma region Operators

// Required operators
vec3 operator/(vec3 num, vec3 denom) {
	return vec3(num.x / denom.x, num.y / denom.y, num.z / denom.z);
}

mat4 transpose(const mat4& m) {
	mat4 mTranspose;
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			mTranspose[i][j] = m[j][i];
		}
	}

	return mTranspose;
}

#pragma endregion

#pragma region Shaders

// Vertex Shader in GLSL
const char* vertexSource = R"(
	#version 330
	precision highp float;
	layout(location = 0) in vec2 cVertexPosition;
	out vec2 texcoord;
	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1);
	}
)";

// Fragment Shader in GLSL
const char* fragmentSource = R"(
	#version 330
	precision highp float;
	uniform sampler2D textureUnit;
	in vec2 texcoord;
	out vec4 fragmentColor;
	void main() {
		fragmentColor = texture(textureUnit, texcoord);	
	}
)";

#pragma endregion

#pragma region Constants

const float epsilon = 0.0001f;

#pragma endregion

#pragma region Material

enum MaterialType { ROUGH, REFLECTIVE};

class Material {
public:
	// Type of the material can be rough or reflective, this determines the BRDF model
	MaterialType type;
	vec3 ka, kd, ks, f0;
	float shine;

	Material(MaterialType _type) : type(_type) { }
};

/*
*
*
*/

class RoughMaterial : public Material {
public:
	RoughMaterial(vec3 _kd, vec3 _ks, float _shine) : Material(ROUGH) {
		ka = _kd * M_PI;
		kd = _kd;
		ks = _ks;
	}
};


/*
* A reflective material can be described with
*
*/
class ReflectiveMaterial : Material {
public:
	ReflectiveMaterial(vec3 n, vec3 kappa) : Material(REFLECTIVE) {
		vec3 one = { 1, 1, 1 };
		f0 = ((n - one) * (n - one) + kappa * kappa) / ((n + one) * (n + one) + kappa * kappa);
	}
};

#pragma endregion

/*
* Represents a light beam
*/
class Ray {
	vec3 start, dir;

public:
	Ray(const vec3& _start, const vec3& _dir) : start(_start), dir(normalize(_dir)) { }
	
	vec3 getStart() const { return start; }
	vec3 getDir() const { return dir; }
};

/*
* Represents an object-ray intersection
*/
class Hit {
	vec3 position, normal;
	Material* material;

public:
	float t;	// Ray parameter (ray(t) = start + dir * t)

	/*
	* Default constructor sets the ray parameter to -1,
	* meaning that there was no (valuable) intersection
	*/

	Hit() : t(-1) { }

	vec3& getPosition() { return position; }
	vec3& getNormal() { return normal; }
	Material* getMaterial() { return material; }

	void setPosition(const vec3& pos) { position = pos; }
	void setNormal(const vec3& n) { normal = normalize(n); }
	void setMaterial(Material* mat) { material = mat; }
};

class Intersectable {
protected:
	Material* material;

public:
	virtual Hit intersect(const Ray& ray) = 0;
};

class QuadricIntersectable : public Intersectable {
protected:
	mat4 tfQ;				// Symmetric matrix, representing a quadric object (transformed and scaled)

public:
	float f(vec4 r) { return dot(r * tfQ, r); }

	vec3 gradf(vec4 r) {
		vec4 g = r * tfQ * 2.0f;
		return vec3{ g.x, g.y, g.z };
	}
};

class Sphere : public Intersectable {
private:
	vec3 center;
	float radius;

public:
	Sphere(const vec3& _center, float _radius, Material* _material) {
		center = _center;
		radius = _radius;
		material = _material;
	}

	Hit intersect(const Ray& ray) override {
		Hit hit;
		vec3 dist = ray.getStart() - center;
		float a = dot(ray.getDir(), ray.getDir());
		float b = dot(dist, ray.getDir()) * 2.0f;
		float c = dot(dist, dist) - radius * radius;
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;

		hit.t = (t2 > 0) ? t2 : t1;
		hit.setPosition(ray.getStart() + ray.getDir() * hit.t);
		hit.setNormal((hit.getPosition() - center) * (1.0f / radius));
		hit.setMaterial(material);

		return hit;
	}
};

class Ellipsoid : public QuadricIntersectable {
private:

public:
	Ellipsoid(const vec3& center, const vec3& params, Material* _material) {

		material = _material;

		mat4 Q = { {1 / (params.x * params.x), 0, 0, 0},
				   {0, 1 / (params.y * params.y), 0, 0},
				   {0, 0, 1 / (params.z * params.z), 0},
				   {0, 0, 0, -1}
		};

		mat4 TInv = { {1, 0, 0, 0},
					  {0, 1, 0, 0},
					  {0, 0, 1, 0},
					  {-center.x, -center.y, -center.z, 1}
		};

		mat4 TInvt = { {1, 0, 0, -center.x },
					   { 0, 1, 0, -center.y},
					   { 0, 0, 1, -center.z},
					   { 0, 0, 0, 1}
		};

		tfQ = TInv * Q * TInvt;
	}

	Hit intersect(const Ray& ray) override {
		Hit hit; // Ray parameter t = -1

		vec4 S = { ray.getStart().x, ray.getStart().y, ray.getStart().z, 1 };
		vec4 D = { ray.getDir().x, ray.getDir().y, ray.getDir().z, 0 };

		float a = dot(D * tfQ, D);
		float b = dot(D * tfQ, S) + dot(S * tfQ, D);
		float c = dot(S * tfQ, S);

		float discriminant = b * b - 4.0f * a * c;
		if (discriminant < 0) return hit;

		float t1 = (-b + sqrtf(discriminant)) / (2.0f * a);
		float t2 = (-b - sqrtf(discriminant)) / (2.0f * a);

		if (t1 <= 0) return hit;

		hit.t = (t2 > 0) ? t2 : t1;
		hit.setPosition(ray.getStart() + ray.getDir() * hit.t); // ray(t) = start + dir * t
		hit.setNormal(gradf(vec4{ hit.getPosition().x, hit.getPosition().y, hit.getPosition().z, 1 }));
		hit.setMaterial(material);

		return hit;
	}
};

struct Cylinder : public Intersectable {
	vec4 r;
	vec3 s;
	vec3 t;


	Cylinder(vec3 r, vec3 t, Material* _material) {
		this->r.x = r.x;
		this->r.y = r.y;
		this->r.z = r.z;
		this->r.w = 1;
		this->s = s;
		this->t = t;
		material = _material;
	}

	mat4 Q() {
		return mat4(vec4(1 / (r.x * r.x), 0, 0, 0),
			vec4(0, 0, 0, 0),
			vec4(0, 0, 1 / (r.z * r.z), 0),
			vec4(0, 0, 0, -1));
	}

	mat4 T() {
		return mat4(vec4(1, 0, 0, 0),
			vec4(0, 1, 0, 0),
			vec4(0, 0, 1, 0),
			vec4(t.x, t.y, t.z, 1));
	}

	mat4 Tinv() {
		return mat4(vec4(1, 0, 0, 0),
			vec4(0, 1, 0, 0),
			vec4(0, 0, 1, 0),
			vec4(-t.x, -t.y, -t.z, 1));
	}

	mat4 Tinvt() {
		return mat4(vec4(1 , 0, 0, -t.x),
			vec4(0, 1 , 0, -t.y),
			vec4(0, 0, 1, -t.z),
			vec4(0, 0, 0, 1));
	}

	mat4 M() {
		return Tinv() * Q() * Tinvt();
	}
	/*
	float f(vec4 r) {
		return dot(r * Q(), r);
	}
	*/

	vec3 gradf(vec4 r) {
		vec4 g = r * M() * 2;
		return normalize(vec3(g.x, g.y, g.z));
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		Hit nonhit;

		vec4 D = vec4(ray.getDir().x, ray.getDir().y, ray.getDir().z, 0);
		vec4 S = vec4(ray.getStart().x, ray.getStart().y, ray.getStart().z, 1);
		float a = dot(D * M(), D);
		float b = dot(D * M(), S) + dot(S * M(), D);
		float c = dot(S * M(), S);

		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return nonhit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;    // t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return nonhit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.setPosition(ray.getStart() + ray.getDir() * hit.t);
		if (hit.getPosition().y > 0.2f || hit.getPosition().y < -0.5f) {
			hit.t = t1;
			hit.setPosition(ray.getStart() + ray.getDir() * hit.t);
			if (hit.getPosition().y > 0.2f || hit.getPosition().y < -0.5f) return nonhit;
		}
		hit.setNormal(gradf(vec4(hit.getPosition().x, hit.getPosition().y, hit.getPosition().z, 1)));
		hit.setMaterial(material);
		return hit;
	}
};

class Camera {
private:
	vec3 eye, lookat, right, up;

public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float fov) {
		eye = _eye;
		lookat = _lookat;

		vec3 w = eye - lookat;
		right = normalize(cross(vup, w)) * length(w) * tanf(fov / 2);
		up = normalize(cross(w, right)) * length(w) * tanf(fov / 2);
	}

	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
};

class Light {
private:
	vec3 direction, Le;

public:
	Light(vec3 _direction, vec3 _Le) : direction(normalize(_direction)), Le(_Le) { }

	vec3& getDirection() { return direction; }
	vec3& getLe() { return Le; }
};

class Scene {
private:
	std::vector<Intersectable*> objects;
	std::vector<Light*> lights;
	Camera camera;
	vec3 ambientLight;

public:
	void build() {
		// Create camera
		vec3 eye = vec3{ 0, 0, 2 };
		vec3 vup = vec3{ 0, 1, 0 };
		vec3 lookat = vec3{ 0, 0, 0 };
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		// Create lights
		ambientLight = vec3{ 0.4f, 0.4f, 0.4f };

		// Create materials
		vec3 kd(0.3f, 0.2f, 0.1f);
		vec3 ks(2, 2, 2);
		Material* material = new RoughMaterial(kd, ks, 50);

		// Create objects
		objects.push_back(new Cylinder(vec3(0.2, 0, 1), vec3(1, 0, 0), material));
		objects.push_back(new Ellipsoid(vec3(0, 0, 0), vec3(0.2, 0.5, 0.5), material));
	}

	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.getDir(), bestHit.getNormal()) > 0) bestHit.setNormal(bestHit.getNormal() * (-1));
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {	// for directional lights
		for (Intersectable* object : objects) if (object->intersect(ray).t > 0) return true;
		return false;
	}

	vec3 trace(Ray ray, int depth = 0) {
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return ambientLight;
		vec3 outRadiance = hit.getMaterial()->ka * ambientLight;
		// Lokális illumináció
		for (Light* light : lights) {
			Ray shadowRay(hit.getPosition() + hit.getNormal() * epsilon, light->getDirection());
			float cosTheta = dot(hit.getNormal(), light->getDirection());
			if (cosTheta > 0 && !shadowIntersect(shadowRay)) {	// shadow computation
				outRadiance = outRadiance + light->getLe() * hit.getMaterial()->kd * cosTheta;
				vec3 halfway = normalize(-ray.getDir() + light->getDirection());
				float cosDelta = dot(hit.getNormal(), halfway);
				if (cosDelta > 0) outRadiance = outRadiance + light->getLe() * hit.getMaterial()->ks * powf(cosDelta, hit.getMaterial()->shine);
			}
		}
		return outRadiance;
	}
};

GPUProgram gpuProgram;
Scene scene;

class FullScreenTexturedQuad {
private:
	unsigned int vao; // Vertex Array Object ID
	Texture texture;  // Texture ID

public:
	FullScreenTexturedQuad(const int windowWidth, const int windowHeight, const std::vector<vec4>& image) : texture(windowWidth, windowHeight, image) {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		unsigned int vbo;
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);

		float vertexCoords[] = { -1, -1, 1, -1, 1, 1, -1, 1 };
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, reinterpret_cast<void*>(0));
	}

	void draw() {
		glBindVertexArray(vao);
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	}
};

FullScreenTexturedQuad* fullScreenTextureQuad;


// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	
	scene.build();
	
	std::vector<vec4> image(windowWidth * windowHeight);
	long timeStart = glutGet(GLUT_ELAPSED_TIME);
	scene.render(image);
	long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));

	fullScreenTextureQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	fullScreenTextureQuad->draw();
	glutSwapBuffers();												// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {

	glutPostRedisplay();
}