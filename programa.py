from pyspark.sql import SparkSession 
from pyspark import SparkFiles 
from ultralytics import YOLO 
# Inicializar la sesión de Spark 
spark = SparkSession.builder.appName("YOLOExample").getOrCreate() 
# Ruta al modelo YOLO en HDFS 
ruta_modelo_hdfs = "hdfs://master:9000/user/hadoop/Yolo/best.pt" 
ruta_local="/home/hadoop/proyecto/Yolo/best.pt" 
#Copiar el modelo YOLO al sistema de archivos local del nodo maestro 
spark.sparkContext.addFile(ruta_modelo_hdfs) 
ruta_modelo_local = 'file://'+SparkFiles.get('best.pt') 
# Cargar el modelo YOLO localmente 
modelo = YOLO (ruta_local) 
# Crear una lista de rutas de imágenes que deseas procesar 
ruta_imagen_hdfs = "hdfs://master:9000/user/hadoop/Yolo/imagenes/c1.png" 
# Cargar la imagen directamente desde HDFS 
imagen = spark.read.format("image").option("drop Invalid", True).load(ruta_imagen_hdfs) 
#Realizar inferencia en la imagen 
resultado = modelo (imagen) 
# Guardar los resultados en un archivo de texto en HDFS
resultado.saveAsTextFile("hdfs://master:9000/user/hadoop/Yolo/imagenes/test1.txt") 
#Detener la sesión de Spark 
spark.stop()