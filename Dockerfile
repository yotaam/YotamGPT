FROM openjdk:17
WORKDIR /app
COPY target/gpt-api-1.0-SNAPSHOT-jar-with-dependencies.jar app.jar
COPY gpt2_weights.json .
COPY models/ models/
#important!!
ENV JAVA_OPTS="-Xmx6g"
EXPOSE 8080
CMD ["sh", "-c", "java $JAVA_OPTS -jar app.jar"]
