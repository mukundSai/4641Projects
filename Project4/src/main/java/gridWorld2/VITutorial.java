package gridWorld2;

import burlap.behavior.policy.GreedyQPolicy;
import burlap.behavior.policy.Policy;
import burlap.behavior.policy.PolicyUtils;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.MDPSolver;
import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.auxiliary.StateReachability;
import burlap.behavior.singleagent.planning.Planner;
import burlap.behavior.valuefunction.ConstantValueFunction;
import burlap.behavior.valuefunction.QProvider;
import burlap.behavior.valuefunction.QValue;
import burlap.behavior.valuefunction.ValueFunction;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldTerminalFunction;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
import burlap.domain.singleagent.gridworld.state.GridAgent;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.mdp.auxiliary.DomainGenerator;
import burlap.mdp.core.action.Action;
import burlap.mdp.core.action.SimpleAction;
import burlap.mdp.core.action.UniversalActionType;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.SADomain;
import burlap.mdp.singleagent.model.FactoredModel;
import burlap.mdp.singleagent.model.FullModel;
import burlap.mdp.singleagent.model.RewardFunction;
import burlap.mdp.singleagent.model.TransitionProb;
import burlap.statehashing.HashableState;
import burlap.statehashing.HashableStateFactory;
import burlap.statehashing.simple.SimpleHashableStateFactory;
import burlap.visualizer.Visualizer;

import java.util.*;


public class VITutorial extends MDPSolver implements Planner, QProvider {

    protected Map<HashableState, Double> valueFunction;
    protected ValueFunction vinit;
    protected int numIterations;
    protected DomainGenerator gridWorld;
    protected RewardFunction rf;

    public VITutorial(SADomain domain, double gamma,
                      HashableStateFactory hashingFactory, ValueFunction vinit, int numIterations){
        this.solverInit(domain, gamma, hashingFactory);
        this.vinit = vinit;
        this.numIterations = numIterations;
        this.valueFunction = new HashMap<HashableState, Double>();
        this.rf = ((FactoredModel) domain.getModel()).getRf();

    }

    @Override
    public double value(State s) {
        if(this.model.terminal(s)){
            return rf.reward(s,new SimpleAction(),s);
        }
        Double d = this.valueFunction.get(hashingFactory.hashState(s));
        if(d == null){
            return vinit.value(s);
        }
        return d;
    }

    @Override
    public List<QValue> qValues(State s) {
        List<Action> applicableActions = this.applicableActions(s);
        List<QValue> qs = new ArrayList<QValue>(applicableActions.size());
        for(Action a : applicableActions){
            qs.add(new QValue(s, a, this.qValue(s, a)));
        }
        return qs;
    }

    @Override
    public double qValue(State s, Action a) {

        if(this.model.terminal(s)){
            return rf.reward(s,a,s);
        }

        //what are the possible outcomes?
        List<TransitionProb> tps = ((FullModel)this.model).transitions(s, a);

        //aggregate over each possible outcome
        double q = 0.;
        for(TransitionProb tp : tps){
            //what is reward for this transition?
            double r = tp.eo.r;

            //what is the value for the next state?
            double vp = this.valueFunction.get(this.hashingFactory.hashState(tp.eo.op));

            //add contribution weighted by transition probability and
            //discounting the next state
            q += tp.p * (r + this.gamma * vp);
        }

        return q;
    }

    @Override
    public GreedyQPolicy planFromState(State initialState) {

        HashableState hashedInitialState = this.hashingFactory.hashState(initialState);
        if(this.valueFunction.containsKey(hashedInitialState)){
            return new GreedyQPolicy(this); //already performed planning here!
        }

        //if the state is new, then find all reachable states from it first
        this.performReachabilityFrom(initialState);

        //now perform multiple iterations over the whole state space
        for(int i = 0; i < this.numIterations; i++){
            //iterate over each state
            for(HashableState sh : this.valueFunction.keySet()){
                //update its value using the bellman equation
                this.valueFunction.put(sh, QProvider.Helper.maxQ(this, sh.s()));
            }
        }

        return new GreedyQPolicy(this);

    }

    @Override
    public void resetSolver() {
        this.valueFunction.clear();
    }

    public void performReachabilityFrom(State seedState){

        Set<HashableState> hashedStates = StateReachability.getReachableHashedStates(seedState, this.domain, this.hashingFactory);

        //initialize the value function for all states
        for(HashableState hs : hashedStates){
            if(!this.valueFunction.containsKey(hs)){
                this.valueFunction.put(hs, this.vinit.value(hs.s()));
            }
        }

    }


    public static void main(String [] args){



    }

}