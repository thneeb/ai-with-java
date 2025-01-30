package de.neebs.ai.control.rl;

import lombok.RequiredArgsConstructor;

import java.util.ArrayList;
import java.util.Collections;
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
        double rewardPercentage = 0;
        List<Transition<A, O>> sample = new ArrayList<>();
        List<Transition<A, O>> listWithRewards = list.stream().filter(f -> f.getReward() != 0).toList();
        for (int i = 0; i < Math.min(batchSize * rewardPercentage, listWithRewards.size()); i++) {
            sample.add(listWithRewards.get(RANDOM.nextInt(listWithRewards.size())));
        }
        for (int i = 0; i < Math.min(batchSize * (1 - rewardPercentage), list.size()); i++) {
            sample.add(list.get(RANDOM.nextInt(list.size())));
        }
        Collections.shuffle(sample);
        return sample;
    }


    public List<Transition<A, O>> last(int batchSize) {
        int size = Math.min(batchSize, list.size());
        return list.subList(list.size() - size, list.size());
    }

}
