#version 330
layout(location=0) out vec4 color;

in vec2 v_TexCoord;

uniform sampler2D u_Texture;
uniform float u_time;

struct Object 
{
    int id;
    float d;
};

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

float sdPlane(in float d, in vec3 n, in vec3 p) 
{
    return (d + dot(normalize(n),p));
}

Object map_the_world(in vec3 p)
{
    //float sphere_0 = distance_from_sphere(p, vec3(0.0), 1.0);
    float d = 1001.0;
    int id = -1;
    //vec4 texcolor = texture(u_Texture,vec2(pow(2.718281,(i/20.0)*0.694),0.0));
    float cubeN = 80;

    float uvx=floor((p.x+1)/2*cubeN)/cubeN;
    vec4 texcolor = texture(u_Texture,vec2(uvx,0.0));//vec2(pow(10,(i/20.0)-1)*1.12,0.0));
    
    float n = (texcolor.g+texcolor.r*256)/600;
    n = floor(n*cubeN/2)/(cubeN/2);
    float cubeSize = 1/cubeN/1.25;

    if(p.x> -1 && p.x< 1 && p.y > 0 && p.y < n){
        d = sdBox(vec3(mod(p.x,cubeSize*2.5)-cubeSize,mod(p.y,cubeSize*2.5)-cubeSize,p.z), vec3(cubeSize));//-vec3(0.0,n,0.0)
        id = 0;
        //d = min(sdBox(vec3(mod(p.x,0.05)-0.02,p.y-0.02,p.z)-vec3(0.0,n,0.0), vec3(0.02)),d);
        //d = sdSphere(vec3(mod(p.x,0.05)-0.02,mod(p.y,0.05)-0.02,p.z),vec3(0.0,n,0.0), 0.02);
        }
    else{
        d = sdPlane(0.0,vec3(0.0,0.0,-1.0),p);
        id = 1;
    }

        //d = 0.0001;
        //d = sdSphere(vec3(mod(p.x,0.05)-0.02,mod(p.y,0.05)-0.02,p.z),vec3(0.0,n,0.0), 0.01+sin((u_time+p.x)*3)*0.01);
    //d = min(sdSphere(vec3(mod(p.x,0.05)-0.03,p.y,mod(p.z,0.05)-0.03),vec3(0.0,n,0.0), 0.02),d);
    //d = min(sdSphere(p,vec3(sin(i+u_time)*0.3,n,cos(i+u_time)*0.3), 0.02),d);
    
    

    // Later we might have sphere_1, sphere_2, cube_3, etc...

    return Object(id,d);
}


vec3 calculate_normal(in vec3 p)
{
    const vec3 small_step = vec3(0.001, 0.0, 0.0);

    float gradient_x = map_the_world(p + small_step.xyy).d - map_the_world(p - small_step.xyy).d;
    float gradient_y = map_the_world(p + small_step.yxy).d - map_the_world(p - small_step.yxy).d;
    float gradient_z = map_the_world(p + small_step.yyx).d - map_the_world(p - small_step.yyx).d;

    vec3 normal = vec3(gradient_x, gradient_y, gradient_z);

    return normalize(normal);
}

vec3 ray_march(in vec3 ro, in vec3 rd)
{
    float total_distance_traveled = 0.0;
    const int NUMBER_OF_STEPS = 16;
    const float MINIMUM_HIT_DISTANCE = 0.001;
    const float MAXIMUM_TRACE_DISTANCE = 1000.0;

    for (int i = 0; i < NUMBER_OF_STEPS; ++i)
    {
        vec3 current_position = ro + total_distance_traveled * rd;

        Object found = map_the_world(current_position);
        float distance_to_closest = found.d;

        if (distance_to_closest < MINIMUM_HIT_DISTANCE) 
        {
            vec3 normal = calculate_normal(current_position);
            vec3 light_position = vec3(2.0, -5.0, 3.0);
            vec3 direction_to_light = normalize(current_position - light_position);

            float diffuse_intensity = max(0.0, dot(normal, direction_to_light));
            if(found.id==0){
                return vec3(v_TexCoord.x,1.0,0.0)*diffuse_intensity;
            }
            if(found.id==1){
                return vec3(0.05,0.05,0.05);
            }
            //vec3(1.0, 0.0, 0.0) * diffuse_intensity;
        }

        if (total_distance_traveled > MAXIMUM_TRACE_DISTANCE)
        {
            break;
        }
        total_distance_traveled += distance_to_closest;
    }
    return vec3(0.05,0.05,0.05);
    //return vec3(0.0);
}



void main(){

    vec2 uv = vec2(v_TexCoord.x,v_TexCoord.y*9/16+0.2) * 2.0 - 1.0;

    vec3 camera_position = vec3(0.0, 0.75, -1.0);
    vec3 ro = camera_position;
    vec3 rd = rotateX(-0.52365)*normalize(vec3(uv, 1.0));//rotateX(sin(u_time))*vec3(uv, 1.0);

    vec3 shaded_color = ray_march(ro, rd);

    color = vec4(shaded_color, 1.0);



}