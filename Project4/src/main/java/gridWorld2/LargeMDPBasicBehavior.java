package gridWorld2;

import burlap.behavior.policy.*;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.auxiliary.StateReachability;
import burlap.behavior.singleagent.auxiliary.performance.LearningAlgorithmExperimenter;
import burlap.behavior.singleagent.auxiliary.performance.PerformanceMetric;
import burlap.behavior.singleagent.auxiliary.performance.TrialMode;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.ValueFunctionVisualizerGUI;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.ArrowActionGlyph;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.LandmarkColorBlendInterpolation;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.PolicyGlyphPainter2D;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.StateValuePainter2D;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.learning.LearningAgentFactory;
import QPackage.QLearning;
import burlap.behavior.singleagent.learning.tdmethods.SarsaLam;
import burlap.behavior.singleagent.planning.Planner;
import burlap.behavior.singleagent.planning.deterministic.DeterministicPlanner;
import burlap.behavior.singleagent.planning.deterministic.informed.Heuristic;
import burlap.behavior.singleagent.planning.deterministic.informed.astar.AStar;
import burlap.behavior.singleagent.planning.deterministic.uninformed.bfs.BFS;
import burlap.behavior.singleagent.planning.deterministic.uninformed.dfs.DFS;
import stochastic.policyiteration.PolicyIteration;
import stochastic.valueiteration.ValueIteration;
import burlap.behavior.valuefunction.ConstantValueFunction;
import burlap.behavior.valuefunction.QProvider;
import burlap.behavior.valuefunction.ValueFunction;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldTerminalFunction;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
import burlap.domain.singleagent.gridworld.state.GridAgent;
import burlap.domain.singleagent.gridworld.state.GridLocation;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.mdp.auxiliary.stateconditiontest.StateConditionTest;
import burlap.mdp.auxiliary.stateconditiontest.TFGoalCondition;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.core.state.State;
import burlap.mdp.core.state.vardomain.VariableDomain;
import burlap.mdp.singleagent.SADomain;
import burlap.mdp.singleagent.common.GoalBasedRF;
import burlap.mdp.singleagent.common.VisualActionObserver;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.model.FactoredModel;
import burlap.mdp.singleagent.model.RewardFunction;
import burlap.mdp.singleagent.oo.OOSADomain;
import burlap.statehashing.HashableStateFactory;
import burlap.statehashing.simple.SimpleHashableStateFactory;
import burlap.visualizer.Visualizer;
import gridWorld1.EXGridState;

import java.awt.*;
import java.util.List;

public class LargeMDPBasicBehavior {

    LargeMDP gwdg;
    SADomain domain;
    RewardFunction rf;
    TerminalFunction tf;
    StateConditionTest goalCondition;
    EXGridState initialState;
    HashableStateFactory hashingFactory;
    SimulatedEnvironment env;



    public LargeMDPBasicBehavior(){
        gwdg = new LargeMDP();
        gwdg.setGoalLocation(new int[] {8,9,9,2,6,7,8,9,12,13,16,17,13,5,6, 6, 6,
                        6, 7, 2, 1, 1, 1, 1, 5, 3, 4, 5, 5, 6,15,17,15,15,16,22,23,
                        24,22,23,24,22,23,24, 2, 3, 4, 2, 3, 4, 2, 3, 4,
                        8, 2,23,12,17, 4, 4, 5, 5, 5, 5, 5, 6, 6, 1, 6,16},
                             new int[] {0,0,1,6,6,6,6,6, 6, 6, 6, 6, 7,9,9,10,11,
                                     12,11,13,15,16,17,18,19,21,21,21,22,22,23,23,
                                     17,16,17,22,22,22,21,21,21,20,20,20,24,24,24,
                                     23,23,23,22,22,22,1, 5, 3, 7, 7,10,11,10,11,
                                     12,13,14,13,14,12,19,16});
        domain = gwdg.generateDomain();

        //more to come...
        initialState = new EXGridState(2,1);
        hashingFactory = new SimpleHashableStateFactory();

        env = new SimulatedEnvironment(domain, initialState);
    }

    public void visualize(String outputPath){
        Visualizer v = gwdg.getVisualizer();
        new EpisodeSequenceVisualizer(v, domain, outputPath);
    }

    public void valueIterationExample(String outputPath){

        Planner planner = new ValueIteration(domain, 0.99, hashingFactory,0.001, 1000);
        Policy p = planner.planFromState(initialState);

        PolicyUtils.rollout(p, initialState, domain.getModel()).write(outputPath + "vi");

        manualValueFunctionVis((ValueFunction)planner, p);

    }

    public void QLearningExample(String outputPath){

        //LearningAgent agent = new QLearning(domain, 0.99, hashingFactory, 0.,1.);

        LearningAgent agent = new QLTutorial(domain, .99, hashingFactory, new ConstantValueFunction(), .9, .1);
        //((QLTutorial)agent).setLearningPolicy(new BoltzmannQPolicy((QLTutorial) agent,30));



        //run learning for 50 episodes
        for(int i = 0; i < 1000; i++){
            Episode e = agent.runLearningEpisode(env);

            e.write(outputPath + "ql_" + i);
            System.out.println(i + ": " + e.maxTimeStep());

            //reset environment for next learning episode
            env.resetEnvironment();
        }

        //testing
        ((QLTutorial)agent).setLearningPolicy(new GreedyQPolicy((QLTutorial) agent));
        for(int i =0; i < 10; i++) {
            Episode e = agent.runLearningEpisode(env);

            e.write(outputPath + "qlTest_" + i);
            System.out.println(i + ": " + e.maxTimeStep());

            //reset environment for next learning episode
            env.resetEnvironment();
        }




    }

    public void policyIterationExample(String outputPath){

        Planner planner = new PolicyIteration(domain, 0.99, hashingFactory, 0.001, 1000, 1000);
        Policy p = planner.planFromState(initialState);

        PolicyUtils.rollout(p, initialState, domain.getModel()).write(outputPath + "pi");
        manualValueFunctionVis((ValueFunction)planner, p);
    }

    public void manualValueFunctionVis(ValueFunction valueFunction, Policy p){

        List<State> allStates = StateReachability.getReachableStates(
                initialState, domain, hashingFactory);

        //define color function
        LandmarkColorBlendInterpolation rb = new LandmarkColorBlendInterpolation();
        rb.addNextLandMark(0., Color.RED);
        rb.addNextLandMark(1., Color.BLUE);

        //define a 2D painter of state values, specifying
        //which variables correspond to the x and y coordinates of the canvas
        StateValuePainter2D svp = new StateValuePainter2D(rb);
        svp.setXYKeys("x", "y",
                new VariableDomain(0, 25), new VariableDomain(0, 25),
                1, 1);

        //create our ValueFunctionVisualizer that paints for all states
        //using the ValueFunction source and the state value painter we defined
        ValueFunctionVisualizerGUI gui = new ValueFunctionVisualizerGUI(allStates, svp, valueFunction);

        //define a policy painter that uses arrow glyphs for each of the grid world actions
        PolicyGlyphPainter2D spp = new PolicyGlyphPainter2D();
        spp.setXYKeys("x", "y",
                new VariableDomain(0, 25), new VariableDomain(0, 25),
                1, 1);

        spp.setActionNameGlyphPainter(LargeMDP.ACTION_NORTH, new ArrowActionGlyph(0));
        spp.setActionNameGlyphPainter(LargeMDP.ACTION_SOUTH, new ArrowActionGlyph(1));
        spp.setActionNameGlyphPainter(LargeMDP.ACTION_EAST, new ArrowActionGlyph(2));
        spp.setActionNameGlyphPainter(LargeMDP.ACTION_WEST, new ArrowActionGlyph(3));
        spp.setRenderStyle(PolicyGlyphPainter2D.PolicyGlyphRenderStyle.DISTSCALED);


        //add our policy renderer to it
        gui.setSpp(spp);
        gui.setPolicy(p);

        //set the background color for places where states are not rendered to grey
        gui.setBgColor(Color.GRAY);

        //start it
        gui.initGUI();


    }

    public static void main(String[] args) {

        LargeMDPBasicBehavior example = new LargeMDPBasicBehavior();
        String outputPath = "output/"; //directory to record results

        //we will call planning and learning algorithms here
        //example.policyIterationExample(outputPath);
        example.valueIterationExample(outputPath);
        //example.QLearningExample(outputPath);

        //run the visualizer
        example.visualize(outputPath);

    }

}
