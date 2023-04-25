from tqdm import tqdm
from NiaPy.task import StoppingTask, OptimizationType


def optimize(benchmark, algorithm, nGEN, num_runs=5):
    best_columns = None
    best_score = 0

    for i in tqdm(range(num_runs)):
        task = StoppingTask(
            D=benchmark.get_length(),
            nGEN=nGEN,
            optType=OptimizationType.MINIMIZATION,
            benchmark=benchmark
        )

        solution_vec, score = algorithm.run(task=task)
        score = 1 - score
        columns = benchmark.select_columns(solution_vec)

        print('--------------')
        print(f'Run {i + 1}')
        print('--------------')
        print(f'Score: {score}')
        print(f'Number of features selected: {len(columns)}\n')
        print('\n')

        if score > best_score:
            best_score = score
            best_columns = columns

    print(f'\nBest score of {num_runs} runs: {best_score}')
    print(f'Number of features selected: {len(best_columns)}')

    return best_columns
