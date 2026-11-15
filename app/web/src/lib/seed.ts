// seed = -1 表示随机：在前端先抽好，三模型对比时三个模型才能共用同一个 seed
export const resolveSeed = (seed: number) => (seed === -1 ? Math.floor(Math.random() * 2 ** 31) : seed)
