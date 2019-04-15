package gridWorld1;

import burlap.behavior.policy.Policy;
import burlap.behavior.policy.PolicyUtils;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.planning.Planner;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.mdp.auxiliary.DomainGenerator;
import burlap.mdp.core.StateTransitionProb;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.core.action.Action;
import burlap.mdp.core.action.UniversalActionType;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.SADomain;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.model.FactoredModel;
import burlap.mdp.singleagent.model.RewardFunction;
import burlap.mdp.singleagent.model.statemodel.FullStateModel;
import burlap.shell.visual.VisualExplorer;
import burlap.statehashing.HashableStateFactory;
import burlap.statehashing.simple.SimpleHashableStateFactory;
import burlap.visualizer.StatePainter;
import burlap.visualizer.StateRenderLayer;
import burlap.visualizer.Visualizer;

import java.awt.*;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Rectangle2D;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;




public class GridWorld implements DomainGenerator {

    public static final String VAR_X = "x";
    public static final String VAR_Y = "y";

    public static final String ACTION_NORTH = "north";
    public static final String ACTION_SOUTH = "south";
    public static final String ACTION_EAST = "east";
    public static final String ACTION_WEST = "west";


    protected int[] goalx = {2, 5, 6};
    protected int[] goaly = {3, 3, 4};


    //ordered so first dimension is x
    protected static int [][] map = new int[][]{
            {0,0,0,0,0,0,0},
            {0,0,1,1,0,0,0},
            {0,0,0,0,1,0,0},
            {0,0,1,0,0,0,0},
            {0,1,0,0,0,0,0},
            {0,1,0,0,0,1,0},
            {0,1,0,0,0,0,0},
    };

    public void setGoalLocation(int[] goalx, int[] goaly){
        this.goalx = goalx;
        this.goaly = goaly;
    }


    @Override
    public SADomain generateDomain() {

        SADomain domain = new SADomain();


        domain.addActionTypes(
                new UniversalActionType(ACTION_NORTH),
                new UniversalActionType(ACTION_SOUTH),
                new UniversalActionType(ACTION_EAST),
                new UniversalActionType(ACTION_WEST));

        GridWorldStateModel smodel = new GridWorldStateModel();
        RewardFunction rf = new ExampleRF(this.goalx, this.goaly);
        TerminalFunction tf = new ExampleTF(this.goalx, this.goaly);

        domain.setModel(new FactoredModel(smodel, rf, tf));

        return domain;
    }


    public StateRenderLayer getStateRenderLayer(){
        StateRenderLayer rl = new StateRenderLayer();
        rl.addStatePainter(new GridWorld.WallPainter());
        rl.addStatePainter(new GridWorld.AgentPainter());


        return rl;
    }

    public Visualizer getVisualizer(){
        return new Visualizer(this.getStateRenderLayer());
    }



    protected class GridWorldStateModel implements FullStateModel{


        protected double [][] transitionProbs;

        public GridWorldStateModel() {
            this.transitionProbs = new double[4][4];
            for(int i = 0; i < 4; i++){
                for(int j = 0; j < 4; j++){
                    double p = i != j ? 0.2/3 : 0.8;
                    transitionProbs[i][j] = p;
                }
            }
        }

        @Override
        public List<StateTransitionProb> stateTransitions(State s, Action a) {

            //get agent current position
            EXGridState gs = (EXGridState)s;

            int curX = gs.x;
            int curY = gs.y;

            int adir = actionDir(a);

            List<StateTransitionProb> tps = new ArrayList<StateTransitionProb>(4);
            StateTransitionProb noChange = null;
            for(int i = 0; i < 4; i++){

                int [] newPos = this.moveResult(curX, curY, i);
                if(newPos[0] != curX || newPos[1] != curY){
                    //new possible outcome
                    EXGridState ns = gs.copy();
                    ns.x = newPos[0];
                    ns.y = newPos[1];

                    //create transition probability object and add to our list of outcomes
                    tps.add(new StateTransitionProb(ns, this.transitionProbs[adir][i]));
                }
                else{
                    //this direction didn't lead anywhere new
                    //if there are existing possible directions
                    //that wouldn't lead anywhere, aggregate with them
                    if(noChange != null){
                        noChange.p += this.transitionProbs[adir][i];
                    }
                    else{
                        //otherwise create this new state and transition
                        noChange = new StateTransitionProb(s.copy(), this.transitionProbs[adir][i]);
                        tps.add(noChange);
                    }
                }

            }


            return tps;
        }

        @Override
        public State sample(State s, Action a) {

            s = s.copy();
            EXGridState gs = (EXGridState)s;
            int curX = gs.x;
            int curY = gs.y;

            int adir = actionDir(a);

            //sample direction with random roll
            double r = Math.random();
            double sumProb = 0.;
            int dir = 0;
            for(int i = 0; i < 4; i++){
                sumProb += this.transitionProbs[adir][i];
                if(r < sumProb){
                    dir = i;
                    break; //found direction
                }
            }

            //get resulting position
            int [] newPos = this.moveResult(curX, curY, dir);

            //set the new position
            gs.x = newPos[0];
            gs.y = newPos[1];

            //return the state we just modified
            return gs;
        }

        protected int actionDir(Action a){
            int adir = -1;
            if(a.actionName().equals(ACTION_NORTH)){
                adir = 0;
            }
            else if(a.actionName().equals(ACTION_SOUTH)){
                adir = 1;
            }
            else if(a.actionName().equals(ACTION_EAST)){
                adir = 2;
            }
            else if(a.actionName().equals(ACTION_WEST)){
                adir = 3;
            }
            return adir;
        }


        protected int [] moveResult(int curX, int curY, int direction){

            //first get change in x and y from direction using 0: north; 1: south; 2:east; 3: west
            int xdelta = 0;
            int ydelta = 0;
            if(direction == 0){
                ydelta = 1;
            }
            else if(direction == 1){
                ydelta = -1;
            }
            else if(direction == 2){
                xdelta = 1;
            }
            else{
                xdelta = -1;
            }

            int nx = curX + xdelta;
            int ny = curY + ydelta;

            int width = GridWorld.this.map.length;
            int height = GridWorld.this.map[0].length;

            //make sure new position is valid (not a wall or off bounds)
            if(nx < 0 || nx >= width || ny < 0 || ny >= height ||
                    GridWorld.this.map[nx][ny] == 1){
                nx = curX;
                ny = curY;
            }


            return new int[]{nx,ny};

        }
    }



    public class WallPainter implements StatePainter {

        public void paint(Graphics2D g2, State s, float cWidth, float cHeight) {

            //walls will be filled in black
            g2.setColor(Color.BLACK);

            //set up floats for the width and height of our domain
            float fWidth = GridWorld.this.map.length;
            float fHeight = GridWorld.this.map[0].length;

            //determine the width of a single cell
            //on our canvas such that the whole map can be painted
            float width = cWidth / fWidth;
            float height = cHeight / fHeight;

            //pass through each cell of our map and if it's a wall, paint a black rectangle on our
            //cavas of dimension widthxheight
            for(int i = 0; i < GridWorld.this.map.length; i++){
                for(int j = 0; j < GridWorld.this.map[0].length; j++){

                    //is there a wall here?
                    if(GridWorld.this.map[i][j] == 1){

                        //left coordinate of cell on our canvas
                        float rx = i*width;

                        //top coordinate of cell on our canvas
                        //coordinate system adjustment because the java canvas
                        //origin is in the top left instead of the bottom right
                        float ry = cHeight - height - j*height;

                        //paint the rectangle
                        g2.fill(new Rectangle2D.Float(rx, ry, width, height));

                    }


                }
            }

        }


    }


    public class AgentPainter implements StatePainter {


        @Override
        public void paint(Graphics2D g2, State s,
                          float cWidth, float cHeight) {

            //agent will be filled in gray
            g2.setColor(Color.GRAY);

            //set up floats for the width and height of our domain
            float fWidth = GridWorld.this.map.length;
            float fHeight = GridWorld.this.map[0].length;

            //determine the width of a single cell on our canvas
            //such that the whole map can be painted
            float width = cWidth / fWidth;
            float height = cHeight / fHeight;

            int ax = (Integer)s.get(VAR_X);
            int ay = (Integer)s.get(VAR_Y);

            //left coordinate of cell on our canvas
            float rx = ax*width;

            //top coordinate of cell on our canvas
            //coordinate system adjustment because the java canvas
            //origin is in the top left instead of the bottom right
            float ry = cHeight - height - ay*height;

            //paint the rectangle
            g2.fill(new Ellipse2D.Float(rx, ry, width, height));


        }



    }


    public static class ExampleRF implements RewardFunction {

        int goalX[];
        int goalY[];
        int[] rewards = {-50, -17, 50};

        public ExampleRF(int[] goalX, int[] goalY){
            this.goalX = goalX;
            this.goalY = goalY;
        }

        @Override
        public double reward(State s, Action a, State sprime) {

            int ax = (Integer)s.get(VAR_X);
            int ay = (Integer)s.get(VAR_Y);

            //are they at goal location?
            for (int i = 0; i < this.goalX.length; i++) {
                if (ax == this.goalX[i] && ay==this.goalY[i]) {
                    return rewards[i];
                }
            }

            if (ax == 5 && ay == 6) {
                return -7;
            }
            if (ax == 6 && ay == 5) {
                return -2;
            }
            if (ax == 2 && ay == 2) {
                return 3;
            }

            return -.01;
        }


    }

    public static class ExampleTF implements TerminalFunction {

        int goalX[];
        int goalY[];

        public ExampleTF(int[] goalX, int[] goalY){
            this.goalX = goalX;
            this.goalY = goalY;
        }

        @Override
        public boolean isTerminal(State s) {

            //get location of agent in next state
            int ax = (Integer)s.get(VAR_X);
            int ay = (Integer)s.get(VAR_Y);

            //are they at goal location?
            for(int i =0; i < this.goalX.length; i++) {
                if (ax == this.goalX[i] && ay == this.goalY[i]) {
                    return true;
                }
            }

            return false;
        }



    }

//    public static void valueIterate() {
//
//        GridWorld gen = new GridWorld();
//        SADomain domain = gen.generateDomain();
//
//        State initialState = new EXGridState(1, 0);
//
//        final HashableStateFactory hashingFactory = new SimpleHashableStateFactory();
//
//        System.out.println("Starting VI");
//        Planner planner = new ValueIteration(domain, 0.9999, hashingFactory, 0.0001, 1000);
//        ((ValueIteration) planner).performReachabilityFrom(initialState);
//        System.out.println("Running VI");
//        ((ValueIteration) planner).runVI();
//
//        System.out.println("Starting planning");
//
//        Policy p = planner.planFromState(initialState);
//
//        System.out.println("Done planning");
//
//        SimulatedEnvironment env = new SimulatedEnvironment(domain, initialState);
//
//        System.out.println("Starting episodes");
//        LinkedList<Episode> episodes = new LinkedList<Episode>();
//
//        // 5 episodes
//        for (int i = 0; i < 5; i++) {
//            System.out.println("\n\nOn episode " + (i + 1));
//
//            Episode curr = PolicyUtils.rollout(p, initialState, domain.getModel());
//
//            episodes.addLast(curr);
//
//            System.out.println("Actions taken: " + curr.actionSequence);
//            System.out.println("Last reward: " + curr.rewardSequence);
//
//            env.resetEnvironment();
//
//        }
//
//        // get actions at each reachable and non-terminal state
//        for (int i = 0; i < GridWorld.map.length; i++) {
//            for (int j = 0; j < GridWorld.map[0].length; j++) {
//                if (GridWorld.map[i][j] != 1 && !domain.getModel().terminal(new ExGridAgent(i, j))) {
//                    // implies not a wall
//                    System.out.println(String.format("At position (%d, %d) the policy action is: %s",
//                            i, j, p.action(new EXGridState(i, j))));
//                }
//            }
//        }
//
//        VisualExplorer exp = new VisualExplorer(domain, env, v);
//
//        exp.addKeyAction("w", ACTION_NORTH, "");
//        exp.addKeyAction("s", ACTION_SOUTH, "");
//        exp.addKeyAction("d", ACTION_EAST, "");
//        exp.addKeyAction("a", ACTION_WEST, "");
//
//        exp.initGUI();
//
//        System.out.println("Finished");
//
//    }



    public static void main(String [] args){

        GridWorld gen = new GridWorld();
        gen.setGoalLocation(new int[] {6, 5, 2}, new int[] {4, 3, 3});
        SADomain domain = gen.generateDomain();
        State initialState = new EXGridState(2, 1);
        SimulatedEnvironment env = new SimulatedEnvironment(domain, initialState);

        Visualizer v = gen.getVisualizer();
        VisualExplorer exp = new VisualExplorer(domain, env, v);

        exp.addKeyAction("w", ACTION_NORTH, "");
        exp.addKeyAction("s", ACTION_SOUTH, "");
        exp.addKeyAction("d", ACTION_EAST, "");
        exp.addKeyAction("a", ACTION_WEST, "");

        exp.initGUI();


    }

}

