from pyspark.sql import SparkSession
from ultralytics import YOLO

# Inicializar la sesión de Spark
spark = SparkSession.builder.appName("YOLOExample").getOrCreate()

# Cargar el modelo YOLO en cada nodo del clúster
model = YOLO('hdfs://master:9000/user/hadoop/Yolo/best.pt')

# Crear una lista de rutas de imágenes que deseas procesar
imagen_paths = ['hdfs://master:9000/user/hadoop/Yolo/imagenes/c1.png']

# Distribuir la lista de rutas como un RDD
imagen_paths_rdd = spark.sparkContext.parallelize(imagen_paths)

# Aplicar el modelo YOLO a cada imagen en paralelo
results_rdd = imagen_paths_rdd.map(lambda path: model(path))

# Recopilar los resultados en el nodo maestro (puede ser costoso si hay muchos resultados)
results = results_rdd.collect()

# Imprimir los resultados
for result in results:
    print(result)

# Guardar los resultados en un archivo de texto en HDFS
results_rdd.saveAsTextFile('hdfs://master:9000/user/hadoop/Yolo/resultados/')

# Detener la sesión de Spark
spark.stop()
