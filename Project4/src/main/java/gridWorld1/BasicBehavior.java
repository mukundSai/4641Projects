package gridWorld1;

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
import burlap.mdp.core.action.SimpleAction;
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

import java.awt.*;
import java.util.List;


public class BasicBehavior {



    GridWorld gwdg;
    SADomain domain;
    RewardFunction rf;
    TerminalFunction tf;
    StateConditionTest goalCondition;
    EXGridState initialState;
    HashableStateFactory hashingFactory;
    SimulatedEnvironment env;



    public BasicBehavior(){
        gwdg = new GridWorld();
        gwdg.setGoalLocation(new int[] {2, 5, 6}, new int[] {3, 3, 4});
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

        Planner planner = new ValueIteration(domain, 0.99, hashingFactory,0.001, 1000, true);
        Policy p = planner.planFromState(initialState);

        long start = System.currentTimeMillis();
        planner = new ValueIteration(domain, 0.99, hashingFactory,0.001, 1000, false);
        p = planner.planFromState(initialState);
        long end = System.currentTimeMillis();
        System.out.println("Value Iteration Run Time: " + (end-start)/1000.0);

        PolicyUtils.rollout(p, initialState, domain.getModel()).write(outputPath + "vi");

        manualValueFunctionVis((ValueFunction)planner, p);

    }

    public void QLearningExample(String outputPath){

        //LearningAgent agent = new QLearning(domain, 0.99, hashingFactory, 0.,1.);

        LearningAgent agent = new QLTutorial(domain, .99, hashingFactory, new ConstantValueFunction(), .7, .1);
        //((QLTutorial)agent).setLearningPolicy(new BoltzmannQPolicy((QLTutorial) agent,30));
        ((QLTutorial)agent).setLearningPolicy(new GreedyQPolicy((QLTutorial) agent));



        //run learning for 50 episodes
        System.out.println("Episode Number, Average Reward, Num Steps");
        for(int i = 0; i < 10000; i++){
            Episode e = agent.runLearningEpisode(env);
            printIterationData(i, (QLTutorial) agent);
            e.write(outputPath + "ql_" + i);
            //System.out.println(i + ": " + e.maxTimeStep());

            //reset environment for next learning episode
            env.resetEnvironment();
        }

        //testing
        ((QLTutorial)agent).setLearningPolicy(new GreedyQPolicy((QLTutorial) agent));
        for(int i =0; i < 10; i++) {
            Episode e = agent.runLearningEpisode(env);

            e.write(outputPath + "qlTest_" + i);
            //System.out.println(i + ": " + e.maxTimeStep());

            //reset environment for next learning episode
            env.resetEnvironment();
        }

        manualValueFunctionVis((ValueFunction)agent, new GreedyQPolicy((QLTutorial)agent));




    }
    public void printIterationData(int i, QLTutorial agent) {
        double numTrials = 5;
        double totalReward = 0;
        int totalSteps = 0;
        for (int j = 0; j < numTrials; j++) {
            Episode episode = PolicyUtils.rollout(new GreedyQPolicy(agent), new EXGridState(2, 1), domain.getModel());
            double sum = 0;
            for (Double reward : episode.rewardSequence) {
                sum += reward;
                //System.out.print(reward + " ");
            }
            EXGridState finalState = ((EXGridState) episode.state(episode.rewardSequence.size()));
            sum += ((FactoredModel) domain.getModel()).getRf().reward(finalState, new SimpleAction(), finalState);
            totalReward += sum;
            totalSteps += episode.rewardSequence.size();
        }
        System.out.println(i + "," + totalReward/numTrials + "," + totalSteps/numTrials);
    }



    public void policyIterationExample(String outputPath){

        Planner planner = new PolicyIteration(domain, 0.99, hashingFactory, 0.001, 1000, 1000, true);
        Policy p = planner.planFromState(initialState);

        long start = System.currentTimeMillis();
        planner = new PolicyIteration(domain, 0.99, hashingFactory,0.001, 1000, 1000, false);
        p = planner.planFromState(initialState);
        long end = System.currentTimeMillis();
        System.out.println("Value Iteration Run Time: " + (end-start)/1000.0);

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
                new VariableDomain(0, 7), new VariableDomain(0, 7),
                1, 1);

        //create our ValueFunctionVisualizer that paints for all states
        //using the ValueFunction source and the state value painter we defined
        ValueFunctionVisualizerGUI gui = new ValueFunctionVisualizerGUI(allStates, svp, valueFunction);

        //define a policy painter that uses arrow glyphs for each of the grid world actions
        PolicyGlyphPainter2D spp = new PolicyGlyphPainter2D();
        spp.setXYKeys("x", "y",
                new VariableDomain(0, 7), new VariableDomain(0, 7),
                1, 1);

        spp.setActionNameGlyphPainter(GridWorld.ACTION_NORTH, new ArrowActionGlyph(0));
        spp.setActionNameGlyphPainter(GridWorld.ACTION_SOUTH, new ArrowActionGlyph(1));
        spp.setActionNameGlyphPainter(GridWorld.ACTION_EAST, new ArrowActionGlyph(2));
        spp.setActionNameGlyphPainter(GridWorld.ACTION_WEST, new ArrowActionGlyph(3));
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

        BasicBehavior example = new BasicBehavior();
        String outputPath = "output/"; //directory to record results

        //we will call planning and learning algorithms here
        example.policyIterationExample(outputPath);
        example.valueIterationExample(outputPath);
        example.QLearningExample(outputPath);

        //run the visualizer
        example.visualize(outputPath);

    }


}
