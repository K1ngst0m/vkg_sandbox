#ifndef CAMERA_H_
#define CAMERA_H_

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

class Camera{
private:
    float fov;
    float znear, zfar;

    void updateViewMatrix();
public:
    enum CameraType {lookat, firstperson};
    CameraType type = CameraType::lookat;

    glm::vec3 rotation = glm::vec3();
    glm::vec3 position = glm::vec3();
    glm::vec4 viewPos = glm::vec4();

    float rotationSpeed = 1.0f;
    float movementSpeed = 1.0f;

    bool updated = false;
    bool flipY = false;

    struct{
        glm::mat4 perspective;
        glm::mat4 view;
    } matrices;

    struct{
        bool left = false;
        bool right = false;
        bool up = false;
        bool down = false;
    } keys;

    bool moving(){return keys.left || keys.right || keys.up || keys.down;}

    float getNearClip(){return znear;}
    float getFarClip(){return zfar;}

    void setPerspective(float fov, float aspect, float znear, float zfar);
    void updateAspectRatio(float aspect);
    void setPosition(glm::vec3 position);
    void setRotation(glm::vec3 rotation);
    void rotate(glm::vec3 delta);
    void setTranslation(glm::vec3 translation);
    void translate(glm::vec3 delta);
    void setRotationSpeed(float rotationSpeed);
    void setMovementSpeed(float movementSpeed);

    void update(float deltaTime);
};

#endif // CAMERA_H_
