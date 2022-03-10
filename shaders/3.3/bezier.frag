#version 330 core
layout(location=0) out vec4 color;

in vec2 v_TexCoord;
varying vec3 v_position;
varying vec3 v_normal;
varying vec3 lightPos;

uniform sampler2D u_Texture;



void main(){
    
    vec4 texColor = texture(u_Texture, v_TexCoord);
    /*
    //vec3 lightPos = vec3(0.0,4.0,3.0);

     vec3 lightDir = normalize(lightPos-v_position);
      vec3 norm = v_normal;
     
      float diff = max(dot(norm, lightDir), 0.0);
    
    color = vec4(texColor.xyz*(0.3+diff),1.0);
    //color = vec4(0.8,0.3,0.2,1.0);
    */
    color = vec4(0,0,0,1.0);
}
