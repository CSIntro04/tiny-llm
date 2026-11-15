# 生成样例（grid = progress）

由 `src/eval_prompts.py` 自动生成，最长 200 个新 token，遇到 `<|endoftext|>` 提前结束。`⏎` 表示换行。

## 各模型的自动指标

| run / checkpoint | 类别 | 平均 distinct-2 | 平均生成长度 | 平均生成速度 (tok/s) |
|---|---|---|---|---|
| s100m / ckpt_tok100M.pt | story | 0.893 | 192 | 200 |
| s100m / ckpt_tok100M.pt | expository | 0.769 | 200 | 170 |
| s100m / ckpt_tok300M.pt | story | 0.906 | 167 | 180 |
| s100m / ckpt_tok300M.pt | expository | 0.854 | 200 | 188 |
| s100m / ckpt_tok1000M.pt | story | 0.833 | 176 | 148 |
| s100m / ckpt_tok1000M.pt | expository | 0.899 | 200 | 162 |
| s100m / ckpt_best.pt | story | 0.920 | 200 | 182 |
| s100m / ckpt_best.pt | expository | 0.811 | 200 | 219 |

## 全部样例

### `Once upon a time`

| 模型 / 设置 | seed | distinct-2 | 续写 |
|---|---|---|---|
| s100m / ckpt_tok100M.pt | 42 | 0.89 | , there was a little boy named Timmy. Timmy loved to play with his toy car. He would drive the car around the park where it went. One day, Timmy's mom asked him to clean up his toys. Timmy didn't want to play. He just wanted to play with his toy car, but he didn't know what to do. ⏎  ⏎ Timmy tried to clean his toys, but he just couldn't. He got frustrated and started to cry. His mom came into the kitchen and saw that he had made a mess in his car. Timmy didn't understand why the dirty car was and he thought he was sad. ⏎  ⏎ His mom looked at the car and saw the broken car. She knew he had to c … |
| s100m / ckpt_tok300M.pt | 42 | 0.91 | , there was a little boy named Timmy. Timmy loved to play with his toy cars and trucks. One day, Timmy's mommy gave him a new toy truck. The new one was red and shiny. ⏎  ⏎ Timmy was so happy to have his new toy car. He played with it all day long. When it was time to go inside, Timmy said goodbye to his new car. ⏎  ⏎ As Timmy walked away, he saw some cars passing by. He saw a cat, a bird, and he tried to catch it. But the cat was too fast and accidentally broke its leg. Timmy was sad and cried. ⏎  ⏎ But then, his mom took him to the store to buy a new car. Timmy was so happy and he forgot abo … |
| s100m / ckpt_tok1000M.pt | 42 | 0.83 | , there was a little boy named Timmy. Timmy loved to play with his toy cars and trucks. One day, Timmy's mommy told him a new toy for his birthday. It was a big, red truck that could make sounds and move by itself. Timmy was so excited to play with it. ⏎  ⏎ Timmy's mommy told him that the new toy was a unique car that could make different noises and move by itself. Timmy loved the sound of the car and couldn't wait to show it to his friends at school.  ⏎  ⏎ On the day of the party, Timmy's friends arrived and they all sat down. Timmy's favorite toy was a ball of yarn, which was unique too. Tim … |
| s100m / ckpt_best.pt | 42 | 0.92 | , there was a little boy named Timmy. Timmy loved to play with his toy cars and trucks all day long. One day, Timmy's mom told him it was time to take a bath. Timmy didn't want to stop playing, but he knew he had to listen to his mom. ⏎  ⏎ As Timmy was taking his bath, he noticed that his favorite toy truck was missing. He looked around the room, but he couldn't find it. Timmy started to cry because he was so upset about losing his favorite toy. His mom came to his room and asked him what was wrong. Timmy told her that he lost his favorite toy truck. His mom said, "Don't worry, we'll find it t … |

### `Photosynthesis is the process`

| 模型 / 设置 | seed | distinct-2 | 续写 |
|---|---|---|---|
| s100m / ckpt_tok100M.pt | 42 | 0.77 |  of the process of the human brain, which we will be on a better equation for the development of the human brain. The human brain is not aware of the most important human brain, but also to be sure to give a more positive connection to people than it takes them to take the brain. The human brain is one of the most important aspects of the human brain. ⏎ The brain is a serious issue in the brain, and it gives some effort to live in healthy ways, while the brain and the brain is able to live, and work hard. This is a great deal of all the resources of life. The brain is a matter of self-presclus … |
| s100m / ckpt_tok300M.pt | 42 | 0.85 |  of the process of the biological process that is used in the human body. It is one of the main processes of physical body, where it is not only the process of physical body, but also the process of human body. ⏎ Types of Physiology for Physiology ⏎ Symphyxin-binding enzymes are essential for the physiology of certain diseases. They help with the function of the nervous system, the nervous system, and other organs, including the nervous system. This enzyme is used to fight off diseases and to help prevent infection. Symptoms include drowsiness, fever, and diarrhea. ⏎ Vision, also known as tinn … |
| s100m / ckpt_tok1000M.pt | 42 | 0.90 |  of the process of photosynthesis, and the chemical reaction takes place in the food chain where the food can be exchanged to produce energy. In this process, plants convert carbon dioxide to oxygen and carbon dioxide, and they are converted to oxygen by the plants. ⏎ The major cause of photosynthesis is loss of water vapor. In this process, they produce carbon dioxide, which in turn converts it to oxygen. The carbon dioxide can then be used to form energy in food, to generate electricity, and to make energy with a high level of humidity. ⏎ What is the cause of photosynthesis? ⏎ The causes of  … |
| s100m / ckpt_best.pt | 42 | 0.81 |  of the photosynthesis that converts light energy (solar) into chemical energy (solar or solar), where it forms chemical bonds with carbon dioxide (CO2) and water vapor (polar). In this process, photosynthesis occurs naturally. ⏎ - When light is trapped by the plant, it is trapped in the process by the water vapor (solar) and water vapor (polar). The energy that is trapped in the plant is then used to create light. ⏎ - A photosynthetic plant that is able to produce light from the sun is known as a photosynthesis plant. ⏎ - The photosynthetic plant is known as the “Potato Farming Cell Plant” be … |
