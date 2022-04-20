#version 330 core
attribute vec3 position;
attribute vec2 texCoord;
attribute vec3 normal;
varying vec2 v_TexCoord;
varying vec3 v_position;
varying vec3 v_normal;

uniform vec3 u_from;
uniform vec3 u_to;
uniform mat4 u_VP;
uniform mat4 u_Model;

void main()
{
    float t = abs(position.z/99);

    float curveLength = distance(u_to, u_from);

    vec3 cp1 = vec3(u_to.x-0.7*(u_to.x-u_from.x),u_from.y,0);
    //vec3 cp1 = vec3(u_to.x+0.5*(curveLength)*sign(u_to.x-u_from.x),u_from.y,0);
    
    //cp1 = cp1.zyx;
    vec3 cp2 = vec3(u_from.x+0.3*(u_to.x-u_from.x),u_to.y,0);
    //vec3 cp2 = vec3(u_from.x-0.5*(curveLength)*sign(u_to.x-u_from.x),u_to.y,0);
    
    //cp2 = cp2.zyx;

    vec3 pos = pow(1-t,3)*u_from + 3*pow(1-t,2)*t*cp1 + 3*(1-t)*cp2*pow(t,2) + pow(t,3)*u_to;
    
    
    
    t = abs(position.z/99)-0.1;
    /*
    vec3 cp3 = vec3(u_to.x-0.7*(u_to.x-u_from.x),u_from.y,0);
    //cp1 = cp1.zyx;
    vec3 cp4 = vec3(u_from.x+0.3*(u_to.x-u_from.x),u_to.y,0);
    //cp2 = cp2.zyx;

    vec3 pos2 = pow(1-t,3)*u_from + 3*pow(1-t,2)*t*cp1 + 3*(1-t)*cp2*pow(t,2) + pow(t,3)*u_to;
    
    vec3 norm = pos2-pos;
    norm = normalize(vec3(norm.y,-norm.x,norm.z));
    */

    pos = pos.yzx;
    

    //vec3 pos = mix(u_from, u_to, abs(position.z/100));
    

    gl_Position =  u_VP * u_Model * vec4(pos+vec3(position.x,0,0),1.0);
    v_TexCoord = texCoord;
    v_normal = normalize(normal).xyz;
    v_position = position.xyz;
    
}