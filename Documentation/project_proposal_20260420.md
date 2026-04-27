- Fashion shopping agent project proposal
    - Part 1: Motivation
        
        ## Motivation
        
        当我在商店里逛街，看到喜欢的衣服时，我可能会先拍下来，然后再回头打字问模型有没有更便宜的类似款，或者有没有其他颜色。
        
        但这种交互其实很慢，因为我需要不断停下来输入文字。
        
        如果我能一边看衣服、一边直接说：
        
        - “这件有没有更便宜的？”
        - “有没有黑色？”
        - “帮我找个更正式一点的类似款”
        
        那整个流程会自然很多，也更适合 real-time shopping exploration。
        
        # Part 1: Motivation
        
        When I shop for clothes in real life, I often see an item I like, take a quick photo of it, and later type questions like: *Is there a cheaper version of this?* or *Does it come in another color?* But that workflow is slow and awkward. Every follow-up requires me to stop browsing, type another query, and manually describe the item again.
        
        A more natural shopping experience would let me keep looking at clothes and simply speak as I browse:
        
        - “Is there a cheaper version of this?”
        - “Do they have it in black?”
        - “Can you find something similar but more formal?”
        
        This kind of interaction is faster, more intuitive, and better suited to real-time shopping exploration. Instead of forcing users to translate what they see into typed search keywords, the system should understand both the garment in view and the spoken shopping request directly.
        
        A better interface would support shopping in the way people naturally think and speak. Instead of pausing to type, the user could simply point the camera at an item and ask questions in real time, such as “Is there a cheaper version of this?” or “Can you find something similar in black?” This would make the interaction more fluid, reduce friction during browsing, and enable a more natural form of real-time shopping exploration.
        
        ### In other words,
        
        When people shop for clothes in stores, they may notice a garment they like, compare it with nearby items, and immediately want to ask follow-up questions such as whether it comes in another color, whether a cheaper alternative exists online, or whether there is a similar but more formal version. In practice, however, this workflow is usually broken by the interface: the user has to stop browsing, type a query, manually describe the garment, and repeat the process for every refinement.
        
        A better interface would support shopping in the way people naturally think and speak. Instead of pausing to type, the user could simply point the camera at an item and ask questions in real time, such as “Is there a cheaper version of this?” or “Can you find something similar in black?” This would make the interaction more fluid, reduce friction during browsing, and enable a more natural form of real-time shopping exploration.
        
    - **Part 2: Why speech + vision are both necessary (Justifying our solution was to add audio modality to VLM)**
        
        # Part 2: Why speech + vision are both necessary
        
        ## (justifying why our solution was to add audio modality to a VLM)
        
        ## Why speech is necessary
        
        Speech is necessary because, in a real-time shopping setting, it is faster and more convenient than typed search. When a user is actively browsing, holding items, moving around a store, or quickly comparing multiple garments, stopping to type every question creates unnecessary friction. Voice makes it much easier to ask short, iterative follow-up questions in the moment.
        
        This matters because shopping queries are often not fully specified upfront. Users naturally refine their intent step by step:
        
        - “Is there a cheaper version of this?”
        - “What about in black?”
        - “Actually, something similar but more formal.”
        - “Not this one — the jacket on the left.”
        
        These are easier to say than to type, especially during active browsing. In this setting, speech is not just an extra convenience feature. It is the more natural interface for fast, multi-turn shopping exploration.
        
        ## Why vision is necessary
        
        Vision is necessary because the user’s request is grounded in a specific garment they are currently looking at. The system must understand:
        
        - which item the user is referring to
        - what visual properties it has
        - how it differs from nearby items
        - what visual similarity means for retrieval
        
        Without vision, the system cannot resolve references like “this one,” “that jacket,” or “the dress on the left.” It also cannot extract appearance cues such as color, silhouette, style, or texture. In that case, the problem collapses into a generic shopping chatbot rather than a visual shopping assistant.
        
        ## Our solution: add audio modality to a VLM
        
        The key reason to add audio as a native modality to a VLM is that so the model can jointly interpret a visual scene and a spoken shopping query.  This application requires the model to jointly interpret:
        
        - a visual scene containing one or more garments
        - a spoken query referring to one of them
        - a shopping-related intent expressed conversationally (ideally, I am not sure if our model should do that or some other module in the system will do this)
        
        This is not just ASR plus image understanding in isolation. The value comes from linking **spoken referring expressions** to **visual grounding** in a real shopping workflow. This makes the system suitable for a real shopping workflow, where spoken referring expressions must be linked directly to visual grounding. 
        
        (I added audio modality so the model can directly localize the garment referred to in speech, instead of requiring the user to type or relying on a separate text-first interaction.)
        
    - Part 3 Two levels of Engineering Questions - Overall product goal vs. Model scope
        
        # Part 3: Core Engineering Question
        
        The core engineering question of this project should be framed at **two levels**. These two levels are closely connected, but they should not be collapsed into one. The first level defines the **overall end-to-end application goal**. The second level defines the **specific role of the fine-tuned audio-VLM within that larger system**.
        
        This distinction is important because the end-to-end goal of the product does **not** necessarily mean that the fine-tuned model itself must solve the entire pipeline. Instead, the end-to-end goal should first define what kind of system we want to build, and then guide us in deciding what part of that system is most appropriate to assign to the current audio-extended VLM. That decision will directly shape the system design, and in turn determine the model’s input, output, and fine-tuning target.
        
        ## Part 3.1: Overall picture — what system do we ultimately want to build?
        
        At the application level, the question is:
        
        ### **What should a spoken visual shopping system ultimately do for the user?**
        
        For this project, the overall end-to-end goal is to build a system that takes a user’s spoken query together with the image or video they are currently viewing, and returns useful shopping results such as:
        
        - cheaper alternatives
        - other colors
        - similar products
        - more formal or more casual variants
        
        This defines the **overall picture** of the project: not just speech transcription or garment recognition, but a spoken visual shopping workflow that supports real-time product exploration.
        
        ## Part 3.2: Model scope — what should the fine-tuned model actually do?
        
        Once the overall system goal is clear, the next question is:
        
        ### **Given that end-to-end goal, what is the right role for the fine-tuned audio-VLM?**
        
        This is where we move from product thinking to system design and model task formulation.
        
        The key point is that the end-to-end system does **not** imply that the fine-tuned model itself must perform the full shopping pipeline. For example, the final system may need to return cheaper alternatives, other colors, or similar products, but that does not mean the audio-VLM itself must directly generate product search results or complete recommendation outputs. Those final results may come from downstream retrieval, search, or recommendation modules.
        
        So the model-scope question becomes:
        
        ### **Within the overall spoken shopping system, what part of the problem should the audio-extended VLM be fine-tuned to solve?**
        
        A concrete example helps illustrate this distinction.
        
        Suppose the end-to-end system should answer the question:
        
        > “Is there a cheaper version of this?”
        > 
        
        At the application level, the final result might be:
        
        - a list of cheaper similar products
        - images and prices
        - a short text or voice response summarizing the results
        
        But at the model level, there are several possible ways to define the task:
        
        **Option 1**
        
        The audio-VLM only outputs a bounding box around the garment referred to by “this.”
        
        **Option 2**
        
        The audio-VLM outputs both the bounding box and a coarse intent such as `find_cheaper`.
        
        **Option 3**
        
        The audio-VLM outputs a bounding box plus a normalized query such as:
        
        “Find cheaper alternatives for this beige blazer.”
        
        **Option 4**
        
        The audio-VLM outputs a bounding box plus structured slots for price, color, and style constraints.
        
        All of these can support the same end-to-end application, but they assign different responsibilities to the fine-tuned model. This is exactly why the second level of the engineering question is necessary: the **overall picture stays the same, while the model scope can vary**.
        
    - Part 4: Task Definition
        
        好，下面我们接着写 **Part 4: Task Definition**。
        
        ## Part 4: Task Definition
        
        The goal of this section is to make the different **model scopes** introduced in Part 3 more concrete by clearly specifying:
        
        - what the model inputs are
        - what versions of the model outputs are possible
        - what each output version means
        - the advantages and limitations of each version
        
        At this stage, we do **not** yet expand into the full system design. Instead, we focus on clearly defining the **task formulation space**.
        
        ---
        
        # Part 4: Task Definition
        
        Given the overall end-to-end goal and the distinction between **application-level behavior** and **model-level responsibility**, the next step is to define the task of the fine-tuned audio-VLM more concretely.
        
        The common setting across all task variants is the following:
        
        ## Common input setting
        
        The model receives:
        
        - **an image or video frame** containing one or more garments
        - **a spoken user query** referring to one garment or expressing a shopping-related request about it
        
        Example spoken queries include:
        
        - “Is there a cheaper version of this?”
        - “Do they have this in black?”
        - “Can you find something similar to the jacket on the left?”
        - “Not this one — the white dress behind it.”
        
        The main design question is what the model should produce as output. Different output definitions correspond to different task formulations, different supervision requirements, and different downstream system designs.
        
        ---
        
        ## Version A: Spoken garment grounding
        
        ### Task
        
        The model identifies **which garment the user is referring to** in the image, based on the spoken query.
        
        ### Input
        
        - image or video frame
        - spoken query
        
        ### Output
        
        - **target bounding box**
        
        ### Example
        
        Input:
        
        - image with several garments
        - audio: “Can you find a cheaper version of this blazer?”
        
        Output:
        
        - a bounding box around the referred blazer
        
        ### What this version means
        
        In this formulation, the model’s responsibility is limited to resolving the spoken reference visually. It does not need to infer the full shopping action or generate any search query. Its role is to determine **what item the user means**.
        
        ### Why this version is attractive
        
        - It is the cleanest and most realistic first milestone.
        - It directly tests the value of adding audio modality to a VLM.
        - It is easier to define, supervise, and evaluate than richer output formats.
        - It keeps the model focused on the most clearly multimodal part of the problem: linking spoken reference to visual grounding.
        
        ### Limitation
        
        This version does not tell the rest of the system what the user wants to do with the garment. Downstream modules would still need to infer whether the user wants a cheaper option, another color, a similar item, and so on.
        
        ---
        
        ## Version B: Spoken garment grounding + coarse intent classification
        
        ### Task
        
        The model identifies the referred garment and predicts a **coarse shopping intent**.
        
        ### Input
        
        - image or video frame
        - spoken query
        
        ### Output
        
        - **target bounding box**
        - **intent label**
        
        ### Example intent labels
        
        - `find_cheaper`
        - `find_other_color`
        - `find_similar`
        - `find_more_formal`
        - `find_more_casual`
        
        ### Example output
        
        ```json
        {
          "target_box": [x1, y1, x2, y2],
          "intent": "find_cheaper"
        }
        ```
        
        ### What this version means
        
        This formulation extends grounding with a lightweight semantic interpretation of the spoken request. The model must answer both:
        
        - which garment the user is talking about
        - what kind of shopping action the user wants
        
        ### Why this version is attractive
        
        - It is still relatively simple and structured.
        - It makes the model more tightly aligned with the shopping workflow.
        - It reduces the burden on downstream modules.
        - It gives the system a more explicit bridge between spoken interaction and retrieval behavior.
        
        ### Limitation
        
        The intent space is still coarse. It may not capture compound requests such as:
        
        - “Find something similar but cheaper”
        - “Show me this in black and in a shorter cut”
        
        So while it is stronger than Version A, it still abstracts away a lot of nuance.
        
        ---
        
        ## Version C: Spoken garment grounding + normalized query generation
        
        ### Task
        
        The model identifies the referred garment and generates a **normalized shopping query** that can be passed downstream.
        
        ### Input
        
        - image or video frame
        - spoken query
        
        ### Output
        
        - **target bounding box**
        - **normalized textual query**
        
        ### Example output
        
        ```json
        {
          "target_box": [x1, y1, x2, y2],
          "query": "Find cheaper black alternatives for this blazer."
        }
        ```
        
        ### What this version means
        
        Instead of predicting a fixed intent label, the model rewrites the user’s spoken request into a cleaner, more retrieval-friendly form. This can serve as an interface between the audio-VLM and a downstream search or recommendation engine.
        
        ### Why this version is attractive
        
        - It is more flexible than a small intent label set.
        - It may connect more naturally to text-based retrieval pipelines.
        - It allows the model to capture more nuanced requests without requiring a complex structured schema.
        
        ### Limitation
        
        This version is harder to supervise and evaluate consistently. Because the output is free-form text, there may be many acceptable reformulations for the same spoken request. That makes the task less clean than box-only or box-plus-intent formulations.
        
        ---
        
        ## Version D: Spoken garment grounding + structured shopping constraints
        
        ### Task
        
        The model identifies the referred garment and outputs a structured representation of the shopping request.
        
        ### Input
        
        - image or video frame
        - spoken query
        
        ### Output
        
        - **target bounding box**
        - **intent**
        - **constraint slots**
        
        ### Possible slots
        
        - color
        - price direction
        - style direction
        - fit preference
        - length preference
        
        ### Example output
        
        ```json
        {
          "target_box": [x1, y1, x2, y2],
          "intent": "search_similar_items",
          "constraints": {
            "price": "lower",
            "color": "black",
            "style": "more_formal"
          }
        }
        ```
        
        ### What this version means
        
        This is the most structured formulation. It asks the model not only to ground the visual reference, but also to convert the spoken shopping request into a machine-friendly representation for downstream retrieval.
        
        ### Why this version is attractive
        
        - It is the closest to a full shopping-agent interface.
        - It supports cleaner downstream tool use.
        - It makes the model output highly interpretable and easy to connect to APIs.
        
        ### Limitation
        
        This is also the most demanding formulation. It requires a carefully designed schema, more complex supervision, and higher confidence that the current model can stably learn multiple structured outputs at once.
        
        ---
        
        ## Comparison across task versions
        
        The four versions can be understood as a progression from **minimal multimodal grounding** to **richer shopping-aware parsing**.
        
        ### Version A
        
        Focuses only on the most essential multimodal question:
        
        **what garment is the user referring to?**
        
        ### Version B
        
        Adds a lightweight answer to the next question:
        
        **what broad shopping action do they want?**
        
        ### Version C
        
        Keeps the grounding task but makes the semantic output more flexible by producing a normalized query.
        
        ### Version D
        
        Pushes toward a more fully structured shopping-agent output.
        
        All four formulations are compatible with the same end-to-end application. What changes is not the overall user-facing goal, but the specific responsibility assigned to the fine-tuned model.
        
        ---
        
        ## Why this design space matters
        
        Choosing among these task versions is important because it will determine:
        
        - the supervision format used for fine-tuning
        - the complexity of synthetic data generation
        - the evaluation protocol
        - how much semantic interpretation is handled by the model itself
        - how much work is delegated to downstream modules
        
        A simpler formulation may be more practical and better aligned with the current stage of the model. A richer formulation may be more powerful, but it also places stronger demands on data design and training stability.
        
        ---
        
        ## Recommended initial direction
        
        For a first practical version of the project, the most realistic options are likely:
        
        - **Version A**, where the model performs spoken garment grounding only
        - **Version B**, where the model performs grounding plus coarse intent classification
        
        These two versions strike the best balance between:
        
        - clear multimodal value
        - realistic supervision requirements
        - practical fine-tuning scope
        - usefulness in a larger shopping pipeline
        
        Versions C and D remain valuable as future extensions, especially if synthetic data generation and output formatting can be made reliable enough.
        
        ---
        
        ## Transition to the next section
        
        Once the task formulation is chosen, the next question becomes how to obtain or construct supervision for that task. This is especially important because different task versions require different kinds of labels: bounding boxes, intent labels, normalized queries, or structured slots. The next step is therefore to examine what kinds of synthetic data can realistically support each formulation.
        
    - Part 4(b): Task Definition After meeting with Nima
        - System Task / Product End Goal
            
            ## **Audio-conditioned composed fashion retrieval**
            
            也就是：
            
            - 输入：reference garment image + spoken modification
            - 输出：retrieve target garment
            
            而不是把 proposal 主任务写成：
            
            - spoken object grounding
            - bounding box + intent classification
            - 便宜商品搜索 API
            
            ### demo flow
            
            - 用户选一张 reference image
            - 说一句 audio query，比如：
                - “make it black”
                - “more formal”
                - “longer sleeves”
            - 系统返回 top-k retrieved items
            - sidebar 放 example image + example audio / text
            
        - Fine-tuned VLM Model  Specific Task
            - 2 Methods Summary
                
                ### 方法 A
                
                先把 multimodal query 翻译成语言
                
                再把语言变成向量
                
                再检索
                
                所以是：
                
                **multimodal query**
                
                → **caption**
                
                → **caption embedding**
                
                → **retrieval**
                
                ---
                
                ### 方法 B
                
                直接把 multimodal query 变成 retrieval 向量
                
                直接检索
                
                所以是：
                
                **multimodal query**
                
                → **query embedding**
                
                → **retrieval**
                
                所以相当于：
                
                ## 方法 B 比方法 A 少了一个中间步骤
                
                也就是少了“先生成 caption，再把 caption encode 成向量”这一步。
                
            - Detailed Description
                
                对，你现在最需要先把这两条方法线的**“系统目标”和“模型训练目标”分开想**。不然很容易混掉。
                
                我先给你一句最核心的话：
                
                ## 这两个方法的共同 end goal 都是：
                
                **retrieve target fashion item**
                
                也就是最后系统返回的是：数据库里最符合用户需求的那件衣服，而不是一句 caption，也不是一个 embedding。FACap 这类 CIR 任务本身的定义就是：给定 **reference image + modification text**，去检索 **target image**。你这里要做的是把 modification text 换成 spoken query。([arXiv](https://arxiv.org/abs/2507.07135?utm_source=chatgpt.com))
                
                ---
                
                # 一、先把三层东西彻底分开
                
                你可以把整个系统分成 3 层来理解：
                
                ## 第 1 层：最终产品目标
                
                用户给：
                
                - reference image
                - audio query
                
                系统最后要返回：
                
                - top-k retrieved fashion items
                - 最理想的那个就是 target fashion item
                
                这层是**整个 project 的最终目标**。
                
                ---
                
                ## 第 2 层：你训练的 model 到底负责什么
                
                这里才是 method 的关键。
                
                你的 model 不一定直接输出最终商品。
                
                它可以只负责生成一个**适合检索的中间表示**。
                
                这个中间表示有两种主流思路：
                
                ### 方法 A
                
                输出一个 **target-oriented caption**
                
                ### 方法 B
                
                输出一个 **retrieval embedding**
                
                这两个都能服务同一个 end goal：最后检索出 target item。
                
                ---
                
                ## 第 3 层：数据库检索怎么发生
                
                不管你上面输出 caption 还是 embedding，最后都还会有一个 retrieval step：
                
                - 要么用 caption embed 后去检索
                - 要么直接拿 learned query embedding 去检索
                
                所以：
                
                **最终 retrieve 出 target item** 是系统目标；
                
                **caption / embedding** 是模型为了实现检索而产生的中间结果。
                
                ---
                
                # 二、第一种方法到底怎么做：caption-generation retrieval baseline
                
                这个是**更稳、最适合作为 proposal 主方法**的版本。
                
                ---
                
                ## 1）这个方法的 end goal 是什么？
                
                还是一样：
                
                ## **retrieve target fashion item**
                
                不是停在 caption。
                
                caption 只是中间桥梁。
                
                ---
                
                ## 2）这个方法里，模型训练目标到底是什么？
                
                这里的关键是：
                
                ## 你的 model 训练目标是：
                
                **给定 reference image + audio query，输出一个 target-oriented caption**
                
                也就是说，它不是输出：
                
                - bounding box
                - intent label
                - 最终商品 ID
                
                而是输出一段文字，描述“用户真正想找的目标衣服应该长什么样”。
                
                ---
                
                ## 3）为什么是 output caption，而不是 output caption embedding？
                
                因为在这个 baseline 里，你训练的是一个**生成式模型**，也就是你的 audio-extended VLM。
                
                它最自然的事情是：
                
                - 输入图像和语音
                - 生成文本
                
                所以它最适合做的是：
                
                ### **image + audio -> generated caption**
                
                而不是直接训练成：
                
                ### **image + audio -> embedding vector**
                
                后者更像第二种 research extension。
                
                所以第一种方法里：
                
                - **模型直接输出的是 caption**
                - **caption embedding 是后处理出来的，不是模型直接 supervised output**
                
                这个区分很重要。
                
                ---
                
                ## 4）这个 caption 应该长什么样？
                
                它应该是一个 **target-oriented caption**，意思是：
                
                不是单纯转录 audio，
                
                也不是单纯描述 reference image，
                
                而是把两者融合起来，输出“目标衣服”的描述。
                
                举个例子：
                
                ### 输入
                
                - reference image: 一件白色长袖蕾丝裙
                - audio query: “make it black and a little shorter”
                
                ### 模型输出 caption
                
                - “a black version of the dress with shorter length and lace sleeves”
                
                这个 caption 的作用，是把：
                
                - 原图里的 base garment identity / style
                - 用户语音里的 modification delta
                
                组合到一起。
                
                这正是你 Mentor 在 meeting 里讲的核心：
                
                audio 说的是 **delta**，image 提供的是 base style，最后模型要把两者合成新的描述。
                
                这也和 CIReVL 这类思路很接近：它就是把 composed retrieval 问题转成一种“视觉 + 修改条件 -> 更好的语言表达 -> 再做 retrieval”的路线。([OpenReview](https://openreview.net/forum?id=EDPxCjXzSb&utm_source=chatgpt.com))
                
                ---
                
                ## 5）然后 retrieval 是怎么发生的？
                
                这一步你要单独看。
                
                ### Step A：先离线处理数据库
                
                数据库里的每个 candidate fashion item：
                
                - 有图片
                - 你给它配一个 caption，或者用数据集已有 caption
                - 再用 text embedding model 把这个 caption 编码成向量
                - 存进 vector database / index
                
                所以数据库里存的是：
                
                - item image
                - item caption
                - item caption embedding
                
                ---
                
                ### Step B：推理时
                
                用户输入：
                
                - reference image
                - audio query
                
                你的 VLM 生成：
                
                - target-oriented caption
                
                然后你再用同一个 text embedding model 去编码这个 generated caption，得到：
                
                - **query caption embedding**
                
                然后拿这个 query embedding 去数据库里做 nearest neighbor search，找最接近的 item。
                
                所以第一种方法的完整链条是：
                
                ## **reference image + audio query**
                
                → **VLM generates target caption**
                
                → **caption embedding model encodes it**
                
                → **retrieve nearest item from caption-embedding database**
                
                → **得到 retrieved target fashion item**
                
                这就是完整闭环。
                
                ---
                
                ## 6）所以你看 dataset 时应该关注什么？
                
                如果你准备走这条 baseline，去看 dataset 的时候你最该盯的是这些字段：
                
                ### A. 有没有 triplet
                
                你最想看到的是：
                
                - reference image
                - modification text
                - target image
                
                因为这定义了 composed retrieval 任务本身。FACap 就是这种格式。([arXiv](https://arxiv.org/abs/2507.07135?utm_source=chatgpt.com))
                
                ### B. target image 有没有对应 caption
                
                因为你 baseline 训练时最好有一个 target-oriented textual supervision。
                
                如果数据里没有现成 target caption，你就得想办法：
                
                - 用已有 product description
                - 或者自己 caption target image
                - 或者从 annotation 里构造 normalized target description
                
                ### C. modification text 的质量高不高
                
                因为你之后要把它变成 speech。
                
                如果 modification text 太短、太粗糙，那你的 audio supervision 也会很弱。
                
                FACap 的一个卖点就是它是为 fine-grained fashion CIR 构造的，强调更细粒度的修改描述。([arXiv](https://arxiv.org/abs/2507.07135?utm_source=chatgpt.com))
                
                ### D. candidate pool 是怎么定义的
                
                因为 retrieval evaluation 最后要在 candidate database 里找 target。
                
                你要弄清楚：
                
                - target 是不是唯一正样本
                - evaluation 是全库检索还是子集检索
                - metric 是 Recall@K 还是别的
                
                ### E. 有没有类别和属性信息
                
                比如 color, sleeve, length, style 这些字段。
                
                这些会帮助你做：
                
                - caption normalization
                - error analysis
                - demo filtering
                
                ---
                
                # 三、第二种方法到底怎么做：更 research、更 end-to-end 的 direct contrastive retrieval
                
                这个版本更接近你 Mentor 后半段讲的内容，也是更像真正 retrieval model 的做法。
                
                ---
                
                ## 1）它的 end goal 是什么？
                
                还是一样：
                
                ## **retrieve target fashion item**
                
                这一点完全不变。
                
                ---
                
                ## 2）它和第一种方法的根本区别是什么？
                
                根本区别在于：
                
                ### 第一种方法
                
                先生成 caption，再借助另一个 embedding model 检索
                
                ### 第二种方法
                
                不生成 caption，直接学 retrieval embedding
                
                也就是：
                
                你不再走“语言中转站”，
                
                而是直接学：
                
                - query side embedding
                - target image embedding
                - 让它们在同一个空间里可比较
                
                ---
                
                ## 3）这个方法里，模型训练目标到底是什么？
                
                这里模型训练目标就不是 caption generation 了。
                
                而是：
                
                ## **learn a query embedding that is close to the target image embedding**
                
                更具体地说：
                
                ### query side
                
                输入：
                
                - reference image
                - audio query
                
                输出：
                
                - 一个 dense embedding vector，表示“用户想找的目标衣服”
                
                ### target side
                
                输入：
                
                - target image
                
                输出：
                
                - 一个 dense embedding vector，表示这个候选衣服
                
                训练目标：
                
                - matching pair 距离近
                - non-matching pair 距离远
                
                通常用：
                
                - contrastive loss
                - InfoNCE / in-batch negatives
                - cosine similarity 之类
                
                这和你 Mentor 说的“正样本拉近，负样本拉远，做 cosine similarity”是完全一致的。
                
                而 FACap/FashionBLIP-2 这类 CIR 研究路线本身也是 retrieval-oriented composed representation learning，不只是生成 caption。([arXiv](https://arxiv.org/abs/2507.07135?utm_source=chatgpt.com))
                
                ---
                
                ## 4）这条路线里的“更 end-to-end”到底是什么意思？
                
                这里的 “end-to-end” 不是说整个产品只有一个模型就全干完。
                
                它更准确的意思是：
                
                ## 你的模型直接学会：
                
                **从 multimodal query 到 retrieval space 的映射**
                
                而不是：
                
                先生成文本
                
                再用另一个 text retriever 去检索
                
                所以更 end-to-end 的点在于：
                
                - query representation 是直接学出来的
                - retrieval objective 直接进入训练
                - 最终优化目标和检索任务本身更一致
                
                ---
                
                ## 5）第二种方法的完整链条是什么？
                
                完整流程是：
                
                ## 训练时
                
                输入 triplet：
                
                - reference image
                - spoken modification
                - target image
                
                做两条支路：
                
                ### Query branch
                
                (reference image + audio)
                
                → audio-VLM / fusion model
                
                → query embedding
                
                ### Target branch
                
                (target image)
                
                → image encoder / VLM vision branch / pooled representation
                
                → target embedding
                
                然后：
                
                - 正确配对的 query-target similarity 提高
                - 错误配对的 similarity 降低
                
                ---
                
                ## 6）推理时怎么检索？
                
                推理时：
                
                ### 先离线处理数据库
                
                把所有 candidate target image 都 encode 成 target embeddings，存起来。
                
                ### 在线时
                
                用户输入：
                
                - reference image
                - audio query
                
                模型生成：
                
                - query embedding
                
                然后直接去 embedding database 里 nearest neighbor search，取 top-k。
                
                所以第二种方法的链条是：
                
                ## **reference image + audio query**
                
                → **query embedding**
                
                → **nearest neighbor search against target-image embeddings**
                
                → **retrieved target fashion item**
                
                你看，它比第一种方法少了“生成 caption”这一步。
                
                ---
                
                ## 7）那 Mentor 说的“grab last token / logits”要怎么理解？
                
                这个地方你不用照抄他原话。
                
                因为 meeting 里他是在快速 brainstorm。
                
                真正落到 proposal / implementation 上，我建议你这样理解得更干净：
                
                ### 你真正需要的是：
                
                从 query branch 和 target branch 分别拿到一个**稳定的 representation**
                
                这个 representation 更合理的候选是：
                
                - pooled hidden states
                - [CLS]-like representation
                - adapter / projector head 输出
                - 一个专门的 retrieval head
                
                而不是死板地写“拿 logits 当 embedding”。
                
                所以 proposal 里写的时候你可以写得更 clean：
                
                > We extract a pooled multimodal representation from the query branch and a visual representation from the target-image branch, then train them with contrastive loss.
                > 
                
                这样就保留了 Mentor 的核心思想，但技术表达会更稳。
                
                ---
                
                ## 8）所以你看 dataset 时应该关注什么？
                
                如果你准备以后做第二条更 research 的路线，那你看 dataset 时就要重点看这些：
                
                ### A. triplet supervision 是否清楚
                
                必须要有：
                
                - reference image
                - modification
                - target image
                
                因为你要知道谁和谁是正样本。
                
                ### B. candidate pool 是否够大
                
                contrastive retrieval 更依赖：
                
                - 足够多 negative examples
                - 合理的 retrieval setting
                - batch 里能不能形成有效难负样本
                
                这也是 Mentor 为什么提 large batch / positives / negatives。
                
                ### C. modification 是否足够细粒度
                
                因为如果 modification 太粗，比如只是 “change color”，那 embedding 学到的东西可能比较浅；
                
                而 fashion retrieval 更难的地方恰恰在于：
                
                - sleeve
                - length
                - formality
                - cut
                - fabric-like cues
                - style nuance
                
                FACap 就是专门强调 fine-grained fashion CIR 的数据。([arXiv](https://arxiv.org/abs/2507.07135?utm_source=chatgpt.com))
                
                ### D. 是否有高质量 target candidates
                
                因为 retrieval 学得好不好，很大程度取决于：
                
                - target 跟 ref 的差异是不是恰到好处
                - negatives 是否 visually confusing
                - 同类款式是不是足够丰富
                
                ### E. 训练规模和 data cleanliness
                
                contrastive retrieval 对脏数据更敏感。
                
                如果 triplet 对应关系乱，模型就很容易学坏。
                
                ---
                
                # 四、两种方法你可以怎么一句话区分
                
                ## 方法 1：主方法 / baseline
                
                **train the model to generate a target-oriented caption, then retrieve via caption embeddings**
                
                更稳、更容易落地、更适合作为 proposal main method。
                
                它的模型输出是 **caption**，不是 embedding。embedding 是后处理。
                
                这个 caption 最终服务于 retrieval。
                
                ## 方法 2：extension / research version
                
                **train the model to directly produce a retrieval embedding for the multimodal query**
                
                更 research、更接近 retrieval 本身、更 end-to-end。
                
                它的模型输出核心是 **embedding**，不是 caption。
                
                然后直接在 embedding 空间里找 target image。
                
                ---
                
                # 五、我建议你现在脑子里先固定住的图
                
                你就记这两个流程图。
                
                ## 图 1：baseline
                
                **ref image + audio**
                
                → **VLM generates target caption**
                
                → **embed caption**
                
                → **retrieve from database of caption embeddings**
                
                → **target fashion item**
                
                ## 图 2：research extension
                
                **ref image + audio**
                
                → **query embedding**
                
                **target image database**
                
                → **target image embeddings**
                
                然后：
                
                → **nearest neighbor search in shared embedding space**
                
                → **target fashion item**
                
                ---
                
                # 六、你现在去看 dataset 时，最该问自己的问题
                
                你打开数据后，优先问这 6 个问题：
                
                1. **它是不是 composed retrieval 格式？**
                    
                    有没有 `reference image + modification + target image`？
                    
                2. **modification 是不是足够细？**
                    
                    能不能支持 fashion 里的细粒度变化？
                    
                3. **target 有没有 caption / product description？**
                    
                    如果没有，baseline 的 supervision 怎么构造？
                    
                4. **candidate pool 怎么组织？**
                    
                    最后 retrieval 是在什么集合里检索？
                    
                5. **有没有 metadata 可用于 demo？**
                    
                    比如 category、color、style、price、brand。
                    
                6. **text 转 speech 后会不会自然？**
                    
                    如果 modification text 太书面，做成 spoken query 可能会很假。
                    
                
                ---
                
                # 七、我给你的最终结论
                
                你刚才那个理解已经很接近正确答案了，我帮你把它修正成最准确的一版：
                
                ## 第一种方法
                
                对，你的 end goal 是 **retrieve target fashion item**。
                
                你训练的 model 的 goal 是 **output caption**，不是直接 output caption embedding。
                
                然后你再用一个 text embedding model 把这个 caption 编码，去 database 里 retrieve item，最后拿到 retrieved target fashion item。
                
                ## 第二种方法
                
                end goal 还是 **retrieve target fashion item**。
                
                但训练目标变成：**直接 output / learn a query embedding**，让它和正确 target image embedding 靠近。
                
                也就是不再通过 caption 做中转，而是直接学 retrieval space。
                
            - Why method B is worthy doing? 而不是只是炫技？
                
                # 那方法 B 真的不只是炫技，而是可能更好吗？
                
                对，**理论上是可能更好的**，不只是炫技。
                
                而且它好，不只是“更酷”，而是有几个很实际的潜在优势。
                
                ---
                
                ## 方法 B 可能更好的原因 1：目标更一致
                
                方法 A 的训练目标是：
                
                > 生成一个好的 caption
                > 
                
                但你的真正任务不是 captioning，
                
                你的真正任务是：
                
                > retrieve the correct target item
                > 
                
                这两件事相关，但**不完全一样**。
                
                有时候一个 caption 看起来挺合理，
                
                但它未必是最适合 retrieval 的表述。
                
                比如：
                
                - “a more formal black version of the dress”
                - “an elegant black dress with similar silhouette”
                
                这两句话语义很接近，
                
                但 retrieval 时，哪句话更好，不一定。
                
                所以方法 A 有点像绕路：
                
                > 先优化“说得对不对”，
                > 
                > 
                > 再希望“说得对”能帮助“找得准”。
                > 
                
                ---
                
                而方法 B 是直接优化：
                
                > 这个 query embedding 能不能把正确 target 拉近、错误 target 拉远
                > 
                
                所以方法 B 的训练目标和最终 retrieval 目标是**直接对齐**的。
                
                这就是它更 research、也更“end-to-end”的本质。
                
                ---
                
                ## 方法 B 可能更好的原因 2：延迟可能更低
                
                你这个直觉也是对的。
                
                ### 方法 A 推理时要做：
                
                1. VLM 生成 caption
                2. 再调用 embedding model 把 caption 编码成向量
                3. 再检索
                
                ### 方法 B 推理时要做：
                
                1. 直接输出 query embedding
                2. 直接检索
                
                所以方法 B 少了一步 text generation / text encoding 的链路，
                
                理论上确实可能：
                
                - latency 更低
                - pipeline 更短
                - engineering 更干净
                
                当然，实际是不是更快，还取决于你怎么实现。
                
                但从系统结构上讲，它确实更直接。
                
                ---
                
                ## 方法 B 可能更好的原因 3：少一个“语言瓶颈”
                
                方法 A 的问题是，它必须先把需求“翻译成文字”。
                
                但有些视觉差异其实很细：
                
                - 更像这件的版型
                - 袖子短一点但不要太紧
                - 材质类似但更正式
                - 保留这个 pattern 但换成 darker tone
                
                这些东西，有时候语言能表达，但不一定表达得最完整。
                
                所以方法 A 有时候会受限于：
                
                ## **language bottleneck**
                
                也就是：
                
                先转成 caption 的过程中，信息可能会损失。
                
                方法 B 则有机会直接在 embedding space 里保留更多“难以语言化但对 retrieval 有用”的信息。
                
                这也是为什么 retrieval 研究里，direct embedding learning 往往是有意义的，不只是花活。
                
                ---
                
                # 但为什么方法 A 还是很值得先做？
                
                因为它**更稳**。
                
                这是 proposal 里非常重要的一点。
                
                ---
                
                ## 方法 A 更稳的原因 1：更符合你现有模型形态
                
                你现在的模型本质上还是一个 audio-extended VLM。
                
                它天然最擅长的是：
                
                - 看图
                - 听音频
                - 输出文本
                
                所以方法 A 是顺着它现在最自然的能力走的。
                
                ---
                
                ## 方法 A 更稳的原因 2：更容易 debug
                
                如果 retrieval 不好，你可以检查：
                
                - caption 生成得对不对
                - embedding model 好不好
                - caption wording 有没有问题
                - database caption 有没有问题
                
                你可以逐层 debug。
                
                但方法 B 一旦 retrieval 不好，你就很难一下子看出来：
                
                - 是 fusion 没学好
                - 是 audio 没对齐
                - 是 contrastive loss 有问题
                - 是 embedding collapse
                - 是 negatives 不够难
                - 是 pooling 不对
                
                所以方法 B 更强，但也更难调。
                
                ---
                
                ## 方法 A 更稳的原因 3：更容易先做出 demo
                
                你 proposal 最重要的不是一上来就 SOTA，
                
                而是先有一个**可信、能做出来、能演示**的版本。
                
                方法 A 很适合这个目的。
                
    - Part 5: Data Synthesis Design