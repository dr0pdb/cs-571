import com.fuzzylite.Engine;
import com.fuzzylite.FuzzyLite;
import com.fuzzylite.Op;
import com.fuzzylite.defuzzifier.Centroid;
import com.fuzzylite.norm.s.Maximum;
import com.fuzzylite.norm.t.AlgebraicProduct;
import com.fuzzylite.rule.Rule;
import com.fuzzylite.rule.RuleBlock;
import com.fuzzylite.term.Trapezoid;
import com.fuzzylite.term.Triangle;
import com.fuzzylite.variable.InputVariable;
import com.fuzzylite.variable.OutputVariable;

class FuzzyController {
    private Engine engine;

    FuzzyController() {
        this.getEngine();
    }

    double process(double curAngle, double curVelocity){

        System.out.println(curAngle+" " + curVelocity);
        StringBuilder status = new StringBuilder();
        if (! engine.isReady(status))
            throw new RuntimeException("[engine error] engine is not ready:n" + status);

        InputVariable velocity = engine.getInputVariable("angularVelocity");
        InputVariable angle = engine.getInputVariable("theta");
        OutputVariable proposedCurrent = engine.getOutputVariable("current");
        velocity.setValue(curVelocity);
        angle.setValue(curAngle);
        engine.process();
        FuzzyLite.logger().info(String.format( "angularVelocity.input = %s and angle.input = %s -> current.output = %s",
                Op.str(curVelocity),Op.str(curAngle), Op.str(proposedCurrent.getValue())));
        return Double.parseDouble(Op.str(proposedCurrent.getValue()));
    }

    private  Engine getEngine() {
        engine = new Engine();
        engine.setName("Inverted Pendulum");
        engine.setDescription("");

        InputVariable angularVelocity = new InputVariable();
        angularVelocity.setName("angularVelocity");
        angularVelocity.setDescription("");
        angularVelocity.setEnabled(true);
        angularVelocity.setRange(-20, 20);
        angularVelocity.setLockValueInRange(true);
        angularVelocity.addTerm(new Trapezoid("negativeSmall", -20, 0));
        angularVelocity.addTerm(new Trapezoid("positiveSmall", 0 , 20));
        angularVelocity.addTerm(new Triangle("zero",-5,5));
        engine.addInputVariable(angularVelocity);

        InputVariable theta = new InputVariable();
        theta.setName("theta");
        theta.setDescription("");
        theta.setEnabled(true);
        theta.setRange(-20, 20);
        theta.setLockValueInRange(true);
        theta.addTerm(new Trapezoid("negativeSmall", -20, 0));
        theta.addTerm(new Trapezoid("positiveSmall", 0 , 20));
        theta.addTerm(new Triangle("zero",-3,3));
        engine.addInputVariable(theta);

        OutputVariable current = new OutputVariable();
        current.setName("current");
        current.setDescription("");
        current.setEnabled(true);
        current.setRange(-30, 30);
        current.setLockValueInRange(true);
        current.setAggregation(new Maximum());
        current.setDefuzzifier(new Centroid(100));
        current.setDefaultValue(0);
        current.setLockPreviousValue(false);
        current.addTerm(new Trapezoid("negativeSmall", -20 , 0));
        current.addTerm(new Trapezoid("positiveSmall", 0 , 20));
        current.addTerm(new Triangle("zero",-3,3));
        current.addTerm(new Trapezoid("negativeMedium",-30,-15));
        current.addTerm(new Trapezoid("positiveMedium",15,30));
        engine.addOutputVariable(current);

        RuleBlock rule = new RuleBlock();
        rule.setName("rule");
        rule.setDescription("");
        rule.setEnabled(true);
        rule.setConjunction(new AlgebraicProduct());
        rule.setImplication(new AlgebraicProduct());
        //rule.setActivation(new General());
        rule.addRule(Rule.parse("if angularVelocity is negativeSmall and theta is negativeSmall then current is positiveMedium", engine));
        rule.addRule(Rule.parse("if angularVelocity is negativeSmall and theta is zero then current is positiveSmall", engine));
        rule.addRule(Rule.parse("if angularVelocity is negativeSmall and theta is positiveSmall then current is zero", engine));
        rule.addRule(Rule.parse("if angularVelocity is zero and theta is negativeSmall then current is positiveSmall", engine));
        rule.addRule(Rule.parse("if angularVelocity is zero and theta is zero then current is zero", engine));
        rule.addRule(Rule.parse("if angularVelocity is zero and theta is positiveSmall then current is negativeSmall", engine));
        rule.addRule(Rule.parse("if angularVelocity is positiveSmall and theta is negativeSmall then current is zero", engine));
        rule.addRule(Rule.parse("if angularVelocity is positiveSmall and theta is zero then current is negativeSmall", engine));
        rule.addRule(Rule.parse("if angularVelocity is positiveSmall and theta is positiveSmall then current is negativeMedium", engine));
        engine.addRuleBlock(rule);
        return engine;
    }
}