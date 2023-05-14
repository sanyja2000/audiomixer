#version 330
layout(location=0) out vec4 color;

in vec2 v_TexCoord;

uniform sampler2D u_Texture;
uniform float u_time;

mat3 rotateX(float theta) {
    float c = cos(theta);
    float s = sin(theta);
    return mat3(
        vec3(1, 0, 0),
        vec3(0, c, -s),
        vec3(0, s, c)
    );
}

float sdSphere(in vec3 p, in vec3 c, float r)
{
    return length(p - c) - r;
}

float sdBox( vec3 p, vec3 b)
{
  vec3 q = abs(p) - b;
  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

float map_the_world(in vec3 p)
{
    float d = 1001.0;
    
    for(int i=0;i<30;i++){
        vec4 texcolor = texture(u_Texture,vec2(pow(10,(i/20.0)-1)*1.12,0.0));
        
        float n = (texcolor.g+texcolor.r*256)/600;
        d = min(sdSphere(p,vec3(sin(i+u_time)*0.3,n,cos(i+u_time)*0.3), 0.02),d);
   
    }

    return d;
}


vec3 calculate_normal(in vec3 p)
{
    const vec3 small_step = vec3(0.001, 0.0, 0.0);

    float gradient_x = map_the_world(p + small_step.xyy) - map_the_world(p - small_step.xyy);
    float gradient_y = map_the_world(p + small_step.yxy) - map_the_world(p - small_step.yxy);
    float gradient_z = map_the_world(p + small_step.yyx) - map_the_world(p - small_step.yyx);

    vec3 normal = vec3(gradient_x, gradient_y, gradient_z);

    return normalize(normal);
}

vec3 ray_march(in vec3 ro, in vec3 rd)
{
    float total_distance_traveled = 0.0;
    const int NUMBER_OF_STEPS = 8;
    const float MINIMUM_HIT_DISTANCE = 0.001;
    const float MAXIMUM_TRACE_DISTANCE = 1000.0;

    for (int i = 0; i < NUMBER_OF_STEPS; ++i)
    {
        vec3 current_position = ro + total_distance_traveled * rd;

        float distance_to_closest = map_the_world(current_position);

        if (distance_to_closest < MINIMUM_HIT_DISTANCE) 
        {
            vec3 normal = calculate_normal(current_position);
            vec3 light_position = vec3(2.0, -5.0, 3.0);
            vec3 direction_to_light = normalize(current_position - light_position);

            float diffuse_intensity = max(0.0, dot(normal, direction_to_light));

            return vec3(v_TexCoord.x,1.0,0.0)*diffuse_intensity;//vec3(1.0, 0.0, 0.0) * diffuse_intensity;
        }

        if (total_distance_traveled > MAXIMUM_TRACE_DISTANCE)
        {
            break;
        }
        total_distance_traveled += distance_to_closest;
    }
    return vec3(0.05,0.05,0.05);
}



void main(){

    vec2 uv = vec2(v_TexCoord.x,v_TexCoord.y*9/16+0.2) * 2.0 - 1.0;

    vec3 camera_position = vec3(0.0, 1.0, -1.0);
    vec3 ro = camera_position;
    vec3 rd = rotateX(-0.5236)*normalize(vec3(uv, 1.0));

    vec3 shaded_color = ray_march(ro, rd);

    color = vec4(shaded_color, 1.0);



}