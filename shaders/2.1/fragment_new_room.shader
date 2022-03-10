
varying vec2 v_TexCoord;
varying vec3 v_position;
varying vec3 v_normal;

uniform sampler2D u_Texture;

void main(){

    vec4 texColor = texture(u_Texture, v_TexCoord);
    float mult = 1.0;
    if(v_position.y>-1.0000001){
      mult = (0.9-v_position.y/2.0);
        if(v_normal.y<0){
        texColor = texture(u_Texture, v_TexCoord)*0.5;
        texColor.z += 0.1;
        mult = 1.0;
      }
    }else{
      mult = (0.7+v_position.y/2.0);
    }
    

    gl_FragColor = vec4(texColor.xyz*mult,1.0);
    //color = vec4(0.8,0.3,0.2,1.0);
}
