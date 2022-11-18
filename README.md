# Recuperacion-Cire2022-UPS-Cuenca
Programa de recuperacion Cire2022 relizado por Equipo UPS Cuenca

En los enlaces se encuentran los videos de la actividad propuesta 
1)Takeshi deberá avanzar 1.3 metros al frente.
2) takeshi debe subir y bajar su brazo robótico
3) takeshi deberá mover el cuello de lado a lado.

Gazebo
https://youtu.be/8YYl5Dv9DJE

Gazebo y RViz
https://youtu.be/hC80xS8lQkU

De igual manera en la carpeta 'Archivos launch' se encuentra el launch de la etapa 5 y un launch donde se ejecuta el mundo y el archivo .py

En la carpeta 'Archivos py' se encuentra el Archivo de correspondiente a esta actividad llamado 'CIRE_2022_UPS_CUENCA.py' que realiza las tres actividades propuestas

En esta imagen se muestra el movimiento de las articulaciones del robot.
![image](https://user-images.githubusercontent.com/112213196/202806166-33a5f2ee-2eaf-48ae-868d-8bd554c59271.png)

En la siguente imagen se muestra como realiza el robot el giro de la cabeza
![image](https://user-images.githubusercontent.com/112213196/202806251-20a93c69-887d-40b3-8a6a-ca71c4dd359f.png)


En los enlaces se ecnuentra el funcionamiento del brazo robótico en donde el robot recorre hacia un punto, luego se usa el planificador de trayectorias del brazo para hacer que recoja un objeto (abriendoi y cerrando la garra) y al final se va hacia la mesa redonda donde debería dejar el objeto.

Planificador de trayectortias - Gazebo
https://youtu.be/qrV6_60vxBQ

Planificador de trayectortias - Gazebo
https://youtu.be/u9MJx_wtK2A

En la carpeta 'Archivos py' se ecnuentra un archivo llamado 'moveit_arm.py' que realiza lo antes detallado.

En esta imagen se muestra como el robot realiza la planificación de la trayectoria para llegar a un punto para recoger un objeto.
![image](https://user-images.githubusercontent.com/112213196/202806052-25cd058c-195f-45e5-af4d-0c969c557a20.png)


En los enlaces de abajo están los videos donde el robot recorre de punto a punto y obtiene las transformadas de las rocas que se encuentran dispersas al rededor de ese punto.

Obtención de transformadas de las rocas de cada punto - Gazebo 
https://youtu.be/pv22YdjKdkA

Obtención de transformadas de las rocas de cada punto - Gazebo  y RViz
https://youtu.be/qtdFVadkj8k

En la carpeta 'Archivos py' se ecnuentra un archivo llamado 'meta_etapa_5_UPS.py' en el cual el robot realiza el recorrido por cada punto.

En las dos imagenes se muestra la transformada que se obtiene de las rocas en dos diferentes puntos.
![image](https://user-images.githubusercontent.com/112213196/202805522-223db9a2-e693-455e-b8f0-f30a7981ab08.png)

![image](https://user-images.githubusercontent.com/112213196/202805868-9354d29c-9799-4f44-a684-93902b0a622e.png)


Todos estos archivos .py fueron ejecutados en el archivo launch de la etapa 5.






