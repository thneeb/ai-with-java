package de.neebs.ai.control.rl;

import lombok.RequiredArgsConstructor;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

@RequiredArgsConstructor
public class ReplayBuffer<A extends Action, O extends Observation> {
    private final List<Transition<A, O>> list = new ArrayList<>();
    private final int capacity;
    private static final Random RANDOM = new Random();

    public void add(Transition<A, O> transition) {
        if (list.size() >= capacity) {
            list.remove(0);
        }
        list.add(transition);
    }

    public List<Transition<A, O>> sample(int batchSize) {
        List<Transition<A, O>> sample = new ArrayList<>();
        for (int i = 0; i < Math.min(batchSize, list.size()); i++) {
            sample.add(list.get(RANDOM.nextInt(list.size())));
        }
        return sample;
    }


}
