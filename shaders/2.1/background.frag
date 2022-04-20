
varying vec2 v_TexCoord;
uniform sampler2D u_Texture;
uniform int u_time;
uniform vec3 xyzoom;
void main(){

    vec3 col = vec3(0.039,0.5,0.4);
    if(mod(v_TexCoord.x-xyzoom.x/4.0,0.1)<0.005){
        col = vec3(1.0,1.0,1.0);
    }
    if(mod(v_TexCoord.y+xyzoom.y/2.0,0.1)<0.005){
        col = vec3(1.0,1.0,1.0);
    }
    gl_FragColor = vec4(col,1.0);
}