
varying vec2 v_TexCoord;
uniform sampler2D u_Texture;
uniform int u_time;
void main(){

    gl_FragColor = vec4(0.0,0.0,0.0,0.6);
    if(v_TexCoord.x<0.01){
        gl_FragColor = vec4(0.3,0.3,0.3,0.6);
    }
}