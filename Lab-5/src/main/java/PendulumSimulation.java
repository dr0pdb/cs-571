/*
    MAIN CODE TO RUN.
    CHANGE gradlew.txt to gradlew.bat
 */


import com.fuzzylite.Op;
import javafx.animation.KeyFrame;
import javafx.animation.Timeline;
import javafx.application.Application;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.paint.Color;
import javafx.scene.shape.Circle;
import javafx.scene.shape.Line;
import javafx.scene.shape.Rectangle;
import javafx.scene.transform.Rotate;
import javafx.stage.Stage;
import javafx.util.Duration;

public class PendulumSimulation extends Application {
    @Override
    public void start(final Stage stage) {
        final Group group = new Group();
        final Scene scene = new Scene(group, 900, 600, Color.WHITE);
        scene.setFill(Color.ALICEBLUE);
        stage.setScene(scene);
        stage.setTitle("Fuzzy Pendulum Controller");
        stage.show();
        //Pendulum Line
        final Line pendulumHand = new Line(0, 175, 0, 0);
        pendulumHand.setTranslateX(450);
        pendulumHand.setTranslateY(350);

        //Pendulum Ball
        final Circle circle = new Circle(0, 0, 10);
        circle.setTranslateX(450);
        circle.setTranslateY(350);
        circle.setFill(Color.BLACK);

        final Rectangle rectangle = new Rectangle(350,525,200,30);
        rectangle.setFill(Color.SIENNA);

        final Label label = new Label("Angular Displacement :");
        label.setLayoutY(5);
        label.setLayoutX(50);

        final TextField theta = new TextField();
        theta.setPromptText("Enter Theta Value");
        theta.setTranslateX(215);
        theta.setTranslateY(5);

        final Label label1 = new Label("Angular Velocity :");
        label1.setTranslateX(450);
        label1.setLayoutY(5);

        final TextField angularVelocity = new TextField();
        angularVelocity.setPromptText("Enter angularVelocity");
        angularVelocity.setTranslateX(570);
        angularVelocity.setTranslateY(5);

        final CheckBox gravity = new CheckBox("Gravity");
        gravity.setTranslateX(50);
        gravity.setLayoutY(275);

        final Button submitInitialConfig = new Button("Submit");
        submitInitialConfig.setTranslateX(770);
        submitInitialConfig.setTranslateY(5);
        submitInitialConfig.setPrefWidth(75);
        submitInitialConfig.setStyle("-fx-background-color: #798ec7");
        final Button pause = new Button("Pause");
        pause.setTranslateX(770);
        pause.setTranslateY(100);
        pause.setPrefWidth(75);
        pause.setStyle("-fx-background-color: #d0604c");
        final TextArea textArea = new TextArea();
        textArea.setTranslateX(50);
        textArea.setPromptText("FuzzyController Output will be displaced");
        textArea.setTranslateY(50);
        textArea.setPrefWidth(700);
        textArea.setPrefHeight(200);
        textArea.setStyle("-fx-text-fill: #0557e3; ");


        final Button play = new Button("Play");
        play.setTranslateX(770);
        play.setTranslateY(50);
        play.setPrefWidth(75);
        play.setStyle("-fx-background-color: #00ff00");

        group.getChildren().add(circle);
        group.getChildren().add(pendulumHand);
        group.getChildren().add(rectangle);
        group.getChildren().addAll(theta,angularVelocity,label,label1,submitInitialConfig,textArea,pause,play,gravity);
        final Timeline[] fiveSecondsWonder = new Timeline[1];
        submitInitialConfig.setOnAction(new EventHandler<ActionEvent>() {
            @Override
            public void handle(ActionEvent actionEvent) {
                FuzzyController fz = new FuzzyController();
                double displacement = Double.parseDouble(theta.getText());
                final Rotate secondRotate = new Rotate(displacement,0,175);

                //moves pendulum hand
                fiveSecondsWonder[0] = new Timeline(new KeyFrame(Duration.seconds(.01), new EventHandler<ActionEvent>() {
                    double velocity = Double.parseDouble(angularVelocity.getText());
                    double displacement = Double.parseDouble(theta.getText());
                    boolean isgravity = gravity.isSelected();
                    @Override
                    public void handle(ActionEvent event) {
                        double angularAcceleration = fz.process(displacement,velocity);
                        if(displacement < 0 && angularAcceleration == 0) angularAcceleration =10;
                        else if(displacement > 0 && angularAcceleration == 0) angularAcceleration = -10;
                        if(isgravity)
                        {
                            if(displacement < 0)
                                angularAcceleration += 9.8*Math.sin(Math.PI*displacement/180);
                            else if(displacement > 0)
                                angularAcceleration -= 9.8*Math.sin(Math.PI*displacement/180);
                        }
                        String x = String.format( "angularVelocity.input = %s and angle.input = %s -> current.output = %s",
                                Op.str(velocity),Op.str(displacement),angularAcceleration);
                        textArea.appendText(x+"\n");
                        displacement = displacement+velocity*0.01+0.5*angularAcceleration*0.01*0.01;
                        velocity = velocity+angularAcceleration*.01;
                        secondRotate.setAngle(displacement);
                    }
                }));
                fiveSecondsWonder[0].setCycleCount(Timeline.INDEFINITE);
                pendulumHand.getTransforms().add(secondRotate);
                circle.getTransforms().add(secondRotate);
                fiveSecondsWonder[0].play();

            }
        });
        pause.setOnAction(new EventHandler<ActionEvent>() {
            @Override
            public void handle(ActionEvent actionEvent) {
                fiveSecondsWonder[0].pause();
            }
        });
        play.setOnAction(new EventHandler<ActionEvent>() {
            @Override
            public void handle(ActionEvent actionEvent) {
                fiveSecondsWonder[0].play();
            }
        });
    }

    public static void main(final String[] arguments) {
        Application.launch(arguments);
    }
}