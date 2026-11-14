# 生成样例（grid = scale）

由 `src/eval_prompts.py` 自动生成，最长 200 个新 token，遇到 `<|endoftext|>` 提前结束。`⏎` 表示换行。

## 各模型的自动指标

| run / checkpoint | 类别 | 平均 distinct-2 | 平均生成长度 | 平均生成速度 (tok/s) |
|---|---|---|---|---|
| s10m / ckpt_best.pt | story | 0.890 | 150 | 354 |
| s10m / ckpt_best.pt | expository | 0.642 | 200 | 375 |
| s50m / ckpt_best.pt | story | 0.908 | 137 | 289 |
| s50m / ckpt_best.pt | expository | 0.728 | 200 | 289 |
| s100m / ckpt_best.pt | story | 0.900 | 148 | 190 |
| s100m / ckpt_best.pt | expository | 0.783 | 178 | 206 |

## 全部样例

### `Once upon a time`

| 模型 / 设置 | seed | distinct-2 | 续写 |
|---|---|---|---|
| s10m / ckpt_best.pt | 42 | 0.90 | , there was a little boy named Timmy. Timmy loved to play with his toy car all day long. One day, Timmy's mommy gave him a big package of a new toy car. Timmy was so happy to show his mommy and daddy. He put on his new toy car and went to see if he could play with it. ⏎  ⏎ As Timmy played with his new toy car, they saw a bird flying by and saw it fly in front of them. Timmy tried to catch it, but he couldn't. He thought about how much he loved his toy car. So, he took it home and showed it to his mommy. She said it was a good toy car and they would play together all day long. ⏎  ⏎ The end. |
| s50m / ckpt_best.pt | 42 | 0.94 | , there was a little boy named Timmy. Timmy loved to play with his toy cars and trucks. One day, Timmy's mommy gave him a big box of blocks and said, "You can build anything you want with this block." ⏎  ⏎ Timmy started building his tower, but then he got a little too close. The tower fell down and the blocks scattered all over the ground. Timmy was sad because he couldn't build anything anymore.  ⏎  ⏎ But then, his friend came over and said, "Don't worry, Timmy. We can build another tower together." They all worked together and built a new tower. It wasn't as big as the first one, but it was  … |
| s100m / ckpt_best.pt | 42 | 0.92 | , there was a little boy named Timmy. Timmy loved to play with his toy cars and trucks all day long. One day, Timmy's mom told him it was time to take a bath. Timmy didn't want to stop playing, but he knew he had to listen to his mom. ⏎  ⏎ As Timmy was taking his bath, he noticed that his favorite toy truck was missing. He looked around the room, but he couldn't find it. Timmy started to cry because he was so upset about losing his favorite toy. His mom came to his room and asked him what was wrong. Timmy told her that he lost his favorite toy truck. His mom said, "Don't worry, we'll find it t … |

### `Lily wanted to`

| 模型 / 设置 | seed | distinct-2 | 续写 |
|---|---|---|---|
| s10m / ckpt_best.pt | 42 | 0.90 |  play with her friends, but they had to be careful and not go near the pond.  ⏎  ⏎ "Go away, hay!" it said. Lily was sad and said, "You are not careful, hay. Go away!" ⏎  ⏎ But she was too lazy to leave the pond. She didn't want to go near the pond. When she tried to climb, she was scared and started to cry. The pond made her jump up and down, but she was too far away.  ⏎  ⏎ But when she was about to give up, she realized what had happened. She quickly ran away and never saw the pond again. From that day on, she never went near the pond again. |
| s50m / ckpt_best.pt | 42 | 0.91 |  play with the hoop too, but she couldn't find one. She asked her mom if they could make a swing hoop for her. Her mom said yes and they went to the store to make the hoop. ⏎  ⏎ When they got to the store, Lily saw a big hoop. She was so happy! She took the hoop and started to play with it. She threw the hoop up in the air and caught it. She even tried to swing it up and down. ⏎  ⏎ But when she was playing, she accidentally threw the hoop too high and it hit the wall. The hoop broke and Lily was sad. Her mom hugged her and said it was okay, accidents happen. They cleaned up the broken hoop and … |
| s100m / ckpt_best.pt | 42 | 0.94 |  play with the ball, but Lily said, "No, it's mine. You have your own toys." ⏎  ⏎ Lily felt sad and went to sit on a bench. She saw Tim and wanted to play with him. She went to him and said, "Please let me play with your ball. I'm gifted too." ⏎  ⏎ Tim felt bad and tried to take the ball from Lily. They started to quarrel and shout. They did not see the man who was watching them from behind the bench. He was angry and came to them. ⏎  ⏎ He shouted, "What are you two doing? You are hurting my children! Get out of here!" ⏎  ⏎ Tim and Lily were scared and dropped the ball. They ran away and hid b … |

### `The little dog was sad because`

| 模型 / 设置 | seed | distinct-2 | 续写 |
|---|---|---|---|
| s10m / ckpt_best.pt | 42 | 0.89 |  he wanted to go home. So, he walked up to the beach and brought a nice sand with him. ⏎  ⏎ The little dog was so happy and he couldn't believe his son. He wanted to go home and give him a hug. He put on his helmet and ran off to find his friend. ⏎  ⏎ At home, the little dog asked, why the dog was so mean. The dog said, "Oh, that's just a bad dog. He can't keep you safe and he can't be so upset." ⏎  ⏎ The little dog felt bad and went off to find the dog. ⏎  ⏎ The little dog was very sad and lonely. He never went back to his friend, but he was also very sad. |
| s50m / ckpt_best.pt | 42 | 0.87 |  he wanted to play, but he knew his mom would be happy. The next day, his mom said they would go to the park to see the big red ball. When they got there, the big red ball was there again and they were both very happy. The big red ball was not lonely anymore. |
| s100m / ckpt_best.pt | 42 | 0.95 |  he wanted to play, but he knew his mommy would be sad too. He tried to be calm and make the dog happy, but it didn't work.  ⏎  ⏎ But then, Timmy had an idea. He remembered that his mommy had called him to come to the park to play with the dog. They played fetch and ran around the park, and Timmy felt happy again. He knew that even if his mommy told him to be calm, sometimes things can be hard to keep going. |

### `One day, Tom and his mom went to`

| 模型 / 设置 | seed | distinct-2 | 续写 |
|---|---|---|---|
| s10m / ckpt_best.pt | 42 | 0.87 |  the park. They saw a big slide, like a plane. Tom wanted to try it, but his mom said no. She said it was not allowed for playing. ⏎  ⏎ Tom and his mom saw Tom's mom sitting on the slide. He looked sad and angry. Tom did not like the plane. He wanted to try it. He shouted, "Tom, Lily, listen to me! I don't want to try it!" ⏎  ⏎ Tom and Lily started to cry. They did not listen to each other. They wanted the plane too. They reached for a slot, but it was too late. They both felt a tang on their faces. They felt sad and scared. ⏎  ⏎ They stopped playing. They looked at the plane and saw that it w … |
| s50m / ckpt_best.pt | 42 | 0.94 |  the beach. They played in the sand and swam in the water. The sun was nice and warm. When it was time to go home, Tom and his mom gave each other a big hug. ⏎  ⏎ Tom said, "Thank you, Mom. I love playing in the sand." His mom smiled and said, "I love playing in the beach, too. It's such a nice day." ⏎  ⏎ The moral of the story is to always listen to your parents and don't be ashamed. It is better to be safe and have fun. |
| s100m / ckpt_best.pt | 42 | 0.86 |  the beach. They wanted to swim in the sea. Tom saw a big wave coming. He wanted to make a big splash. Tom's mom said, "We can try to make a big splash, Tom." So, they found a big stick. Tom used the stick to poke the water. They made a big splash and laughed. ⏎  ⏎ After a while, the big wave was gone. It was very quiet. Tom and his mom took a photo of the big splash. They felt happy. They had made a fun, noisy wave. Tom's mom laughed too. They had a great day at the beach. |

### `There was a big tree in the park`

| 模型 / 设置 | seed | distinct-2 | 续写 |
|---|---|---|---|
| s10m / ckpt_best.pt | 42 | 0.89 | . She was very excited to explore it. ⏎  ⏎ She ran up to the tree and started to climb the tree. She had a lot of fun and soon she was able to reach the top. ⏎  ⏎ She ran around the tree to pick up the branches and branches. She climbed high and down, up and down. She had so much fun! ⏎  ⏎ She kept climbing down the tree, just in no time. She made it to the top of the tree and was proud of herself. She had been able to reach the top! |
| s50m / ckpt_best.pt | 42 | 0.88 |  and the little girl wanted to climb it. So she stepped on a branch and started to climb up. When she stepped on a branch, a big bird flew down and said, "You can't climb that tree! You must be careful!"  ⏎  ⏎ The little girl was scared and started to cry. She didn't know what to do. Then her mum came over and said, "I'll help you, sweetheart." Her mum was very gentle and stern and kind. She let the little girl climb down to safety, and then she gave her a big hug.  ⏎  ⏎ The little girl was very happy and relieved. She said, "Thank you, mum! I'm safe now!" Then she carefully climbed back down  … |
| s100m / ckpt_best.pt | 42 | 0.83 | . One day, a boy and a girl walked up to the tree. They wanted to open the tree. ⏎  ⏎ The boy said, "We can open the tree!" But the girl was scared. ⏎  ⏎ The boy said, "Please don't open it." ⏎  ⏎ But the girl was very brave. She said, "Please, it is not so bad." ⏎  ⏎ So they opened the tree. ⏎  ⏎ The girl smiled. "Thank you," she said. ⏎  ⏎ The boy and girl were very happy to have opened the tree. |

### `Photosynthesis is the process`

| 模型 / 设置 | seed | distinct-2 | 续写 |
|---|---|---|---|
| s10m / ckpt_best.pt | 42 | 0.66 |  of the process of production. This process requires a lot of nutrients to be managed for the production of the products. However, it is not a problem. In addition, the process of production of the process is the process of production and the production of industrial products. So, the process of production is very important. So it is known that the content is not justified. ⏎ In the process of production of products, the process through production is the process of production of the process, it can be processed by making it to process, and then produced. This process is not the process of prod … |
| s50m / ckpt_best.pt | 42 | 0.77 |  of the process of the synthesis of carbon and hydrogen (CO) in food. The carbon is transferred into gases, where it forms a gas as a carbon dioxide molecule. ⏎ What is the Potential of the Effect of Growth on Potential of the Agriculture Process? ⏎ Growth is the process of the production of new crops. The production of new crops, through fertilizer, is the process of the production of new crops. This allows the food to be used in a variety of soil types, as well as the soil organic matter (GHG) content can be transferred up to the production of crops. ⏎ In addition, the increase in productivi … |
| s100m / ckpt_best.pt | 42 | 0.81 |  of the photosynthesis that converts light energy (solar) into chemical energy (solar or solar), where it forms chemical bonds with carbon dioxide (CO2) and water vapor (polar). In this process, photosynthesis occurs naturally. ⏎ - When light is trapped by the plant, it is trapped in the process by the water vapor (solar) and water vapor (polar). The energy that is trapped in the plant is then used to create light. ⏎ - A photosynthetic plant that is able to produce light from the sun is known as a photosynthesis plant. ⏎ - The photosynthetic plant is known as the “Potato Farming Cell Plant” be … |

### `The water cycle describes`

| 模型 / 设置 | seed | distinct-2 | 续写 |
|---|---|---|---|
| s10m / ckpt_best.pt | 42 | 0.56 |  as the water cycle itself. The water cycle and the water cycle are the main cycle of each cycle. When hydroxylene water is not present, the water cycle itself becomes dry and dry, is the cycle of a change in water cycle. ⏎ For example, water cycle begins through the cycle of hydroxylene water cycle. ⏎ If water is present in water, water cycle is present in water cycle. ⏎ For example, water cycle cycle begins through the cycle cycle after the cycle begins to rise after the cycle begins and goes through the cycle cycle. ⏎ Depending on the cycle, water cycle cycle reaches 8 to 10 years. ⏎ - Clim … |
| s50m / ckpt_best.pt | 42 | 0.61 |  how the water cycle of a river starts in and exits on a water column. The rivers in the water column have a similar water cycle. The water cycle is an example of how water flows through the rivers. The water cycle is considered as an example of how water flows through rivers. ⏎ What are the major water cycle ⏎ The water cycle is the main process of the water cycle. Water flows through rivers. Water flows through rivers, streams and streams. Water flows through rivers and streams. There are two main types of water cycle of rivers: ⏎ - River Water cycle ⏎ - Load River cycle ⏎ - Water Type cycle … |
| s100m / ckpt_best.pt | 42 | 0.85 |  how the water cycle, including the amount of rain or the amount of water that is produced by the system, can be measured. This water cycle allows us to assess the water cycle, and also to make predictions based on available data. ⏎ It’s important to note that water is not a single-age thing. Therefore, it is not solely a matter of the water cycle. It is also a matter of the water cycle. |

### `In mathematics, a prime number`

| 模型 / 设置 | seed | distinct-2 | 续写 |
|---|---|---|---|
| s10m / ckpt_best.pt | 42 | 0.45 |  of the following lines of the first line is used in the following lines: ⏎ 1. In sums the first line is a set of a divisement of a divisement of the divisor in the second line and the divisor is an equilibrium of the divisor. ⏎ 3. In sums the divisor is 1 and the divisor is 1, and the divisor is 5. The divisor is 2, and so that is 2, to 2, and so that is 1, a divisor is 2, and so that is 1, and so that is 1, and so that means 2, and so that is 1, and so there is 2. ⏎ 4. In sums the divisor is 1, and so is 1, and so that is 2, and so on. ⏎ 5. In sums the divisor is 2, and the divisor is 2, |
| s50m / ckpt_best.pt | 42 | 0.53 |  is the number of times a number is added to a given number. The prime number is a number of times or number of times it is added to a number. A prime number is a prime number that is added to a number to make it more likely that it will be added to a number. ⏎ A prime number is a number that is added to a number in a number to make it more likely that it will be added to a number. The prime number can include: ⏎ - Prime numbers: 1, 2, 3, 4 ⏎ - Prime number: 9, 9 ⏎ - Prime number: 6, 10, 5, 10, 11 ⏎ - Prime number: 80, 134, 138, 138, 149 ⏎ - Prime number: 6, 5 ⏎ - Prime number: 74, 5, 144, 149 … |
| s100m / ckpt_best.pt | 42 | 0.63 |  is the unit of the length of a single prime number. This is usually expressed as (1,2,3,4)^2. ⏎ 2. The division of the resulting result into two parts is called a multiplication of 2 prime numbers. For example, the multiplication of 3 prime numbers by 2 prime numbers is 2. ⏎ 3. The division of a number into 3 prime numbers is called a prime number. It is a common form of prime number to divide and work out numbers. There are 3 prime numbers, 9 prime numbers, and 10 prime numbers. ⏎ 4. The prime number is a unit of the length of a prime number. It is a common form of prime numbers to divide an … |

### `The French Revolution began`

| 模型 / 设置 | seed | distinct-2 | 续写 |
|---|---|---|---|
| s10m / ckpt_best.pt | 42 | 0.67 |  in the aftermath of the Revolution, and the Roman Empire was the first settlement of the Roman Empire. In the 17th Century, the French Revolution brought new powers to the Roman Republic to the world, with a number of its own. ⏎ The French Revolution was one of the major economic activities in the 17th Centuries. These were the few of the most successful French Revolution. The German Revolution was in the 18th Century. During the 18th Centuries it began with the French Revolution, when the French Revolution was finally formed. This led to the early expansion of the French Revolution. ⏎ Minist … |
| s50m / ckpt_best.pt | 42 | 0.82 |  in the afternoon of 1864, and the French Revolution resulted in the establishment of its own government. Under Louis XIV's rule, France and Spain had to separate from the Catholic world, with France only to be forced to move to the newly independent states. ⏎ The French Revolution also led to the establishment of the Republic of Louis XIV in 1864. In 1864, the French Revolution took place in Paris, and the revolution began in 1861. In 1866, the French Revolution had reached an agreement on the authority of the French Government to protect the government by moving the French people up from the … |
| s100m / ckpt_best.pt | 42 | 0.83 |  in the aftermath of the Revolutionary War, and on 17 September 1789, the French Revolution was declared an international war of war. The French Revolution gave way to the British Royal Declaration of Rights and the British Royal Declaration of Rights. ⏎ England was also known for its religious and military prosperity. In the 15th century, the German Empire was in a state of rapid decline. During the American Revolutionary War, the government declared a state of emergency and was required to respond. This led to the declaration of the British Empire. ⏎ The Declaration of Rights of Man (1793) ⏎ … |

### `To stay healthy, people should`

| 模型 / 设置 | seed | distinct-2 | 续写 |
|---|---|---|---|
| s10m / ckpt_best.pt | 42 | 0.88 |  be given control of the child’s mind and take good care of the child for these children. ⏎ If you or someone is not aware of this, you should get an appointment of your child’s mind, or you should be given control over it. ⏎ Sleeping is a common condition that may be affecting your child’s health. ⏎ Fruits and vegans of the body, such as the nerve, ginge, etc. ⏎ Sleeping is a key factor in adolescence, and it increases your risk of a child’s physical activity. ⏎ Stress is a key component in sleep. It can feel overwhelming, but it can also be a risk factor for an emergency. ⏎ The lack of sleep … |
| s50m / ckpt_best.pt | 42 | 0.90 |  be given the resources, such as food, water, and a healthy diet. ⏎ Take a few minutes each day to eat a variety of foods and drinks and ensure that you know all the foods you are consuming. You do not have to be an adult, have to be a member of the group. ⏎ If you eat too much, it is best to take a few minutes of water. You could be drinking enough water, but this doesn’t mean you don’t need to drink enough food. ⏎ Take a few minutes every day, as you can drink 10 or more of water. You can also drink 30 to 80 of a daily fluid. ⏎ You can also drink a lot of water, and there aren’t many healthy … |
| s100m / ckpt_best.pt | 42 | 0.79 |  be given fresh fruits, vegetables, fruits, and grains. They should also eat these healthy foods in the right way. Also, it is not surprising that most of us are actually eating healthy foods in the right way. ⏎ There are many ways you can support your health in the right way. They should also be able to eat healthy foods which include whole grains, such as dairy, bread, and cereals. It is important to remember to include a healthy amount of fruits and vegetables in your diet to support your health. ⏎ It is also important to understand the consequences of eating too much salt. It is important  … |
