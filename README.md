# NaturalLanguageprocessing1
v# NLP
## 1.	DataSet Танилцуулга
IMDB Movie Review Sentiment Dataset-ийг судлах явцад уг өгөгдлийн багц нь энгийн эерэг, сөрөг ангиллаас давсан олон давхар утга агуулж байгааг ажигласан. Нэгдүгээрт, шүүмжүүдийн текст дотор илэрхийлэгдэж буй сэтгэл хөдлөл нь ихэнх тохиолдолд шугаман бус бүтэцтэй байдаг. Өөрөөр хэлбэл, нэг шүүмжийн эхэнд эерэг хандлага илэрхийлж байгаад төгсгөл хэсэгт сөрөг дүгнэлт гаргах, эсвэл эсрэгээрээ бичих тохиолдол элбэг ажиглагддаг. Энэ нь sentiment analysis-ийг зөвхөн түлхүүр үг илрүүлэх асуудал биш, харин өгүүлбэр хоорондын утга зүйн уялдаа, ерөнхий контекстийг ойлгох асуудал болохыг харуулдаг.
Мөн датасет дэх шүүмжүүдийн урт нь загварын гүйцэтгэлд нөлөөлөх чухал хүчин зүйл болдог. Олон шүүмжүүд харьцангуй урт, задгай хэлбэртэй бичигдсэн байдаг тул энгийн Bag-of-Words эсвэл зөвхөн үгийн давтамжид суурилсан аргууд нь шүүмжийн сүүл хэсэгт агуулагдах бодит дүгнэлтийг хангалттай тусгаж чаддаггүй. Ийм нөхцөлд дараалалд суурилсан (sequence-based) загварууд илүү давуу талтай болох нь ажиглагддаг ч эдгээр загварууд ч мөн урт текстийн эхэн хэсэгт өгөгдсөн мэдээлэлд хэт хамаарах сул талтай байж болох юм.
Судалгааны явцад анхаарал татсан өөр нэг асуудал нь датасетийн шошгололтын онцлог байв. IMDB датасетэд 5–6 оноотой, тодорхой бус хандлагатай шүүмжүүдийг зориудаар хассан нь сургалтын өгөгдлийг илүү “цэвэр” болгож өгдөг боловч бодит хэрэглээний орчинд тулгардаг холимог, хоёрдмол утгатай сэтгэл хөдлөлийг бүрэн төлөөлж чаддаггүй. Иймээс энэхүү датасет дээр өндөр нарийвчлалтай ажилласан загваруудыг бодит хэрэглээнд шууд шилжүүлэхдээ болгоомжтой хандах шаардлагатай гэж үзэж байна.
Түүнчлэн ихэнх оюутны ажлуудад орхигддог unsup буюу шошгогүй өгөгдлийн хэсэг нь судалгааны хувьд чухал ач холбогдолтой байж болохыг тэмдэглэх нь зүйтэй. Уг өгөгдлийг ашиглан хэлний загварыг урьдчилан сургах, эсвэл хагас хяналттай сургалтын арга хэрэглэх боломжтой бөгөөд энэ нь ангиллын загварын ерөнхий ойлголтыг сайжруулах давуу талтай. Энэ нь IMDB датасетийг зөвхөн supervised classification бус, илүү өргөн хүрээний NLP судалгаанд ашиглах боломжтойг харуулж байна.
Эцэст нь, IMDB Movie Review Sentiment Dataset нь албан бус, ярианы хэв маягтай, зарим тохиолдолд ёжлол, хэтрүүлэг, сэтгэл хөдлөлийн огцом шилжилт агуулсан текстүүдээс бүрддэг. Ийм төрлийн өгөгдөл дээр ажиллах нь загварын бодит чадвар, хязгаарыг илрүүлэхэд тустай бөгөөд “өндөр accuracy” гэх тоон үзүүлэлтээс гадна, загвар үнэхээр утгыг ойлгож байна уу гэсэн асуултыг тавихад хүргэдэг. Ийм өнцгөөс авч үзвэл уг датасет нь зөвхөн сургалтын хэрэгсэл бус, машин сургалтын загваруудын ойлголтын гүнзгий байдлыг шалгах шалгуур болж чадна.
## 2.  DATASET АШИГЛАГДСАН БАЙДАЛ
### 2.1 Domain Adaptable Model for Sentiment Analysis

		
•	Товч танилцуулга
Domain adaptable sentiment analysis загвар нь нэг домэйн (жишээ нь кино шүүмж) дээр сурсан мэдлэгээ өөр домэйнд (жишээ нь бүтээгдэхүүний үнэлгээ, сошиал медиа пост) аль болох бага гүйцэтгэлийн уналттайгаар ашиглах зорилготой загвар юм. Энгийн sentiment analysis загварууд нь ихэвчлэн нэг төрлийн өгөгдөлд сайн ажилладаг боловч домэйн солигдох үед (үг хэллэг, хэв маяг, утга агуулга өөрчлөгдөхөд) гүйцэтгэл нь огцом буурдаг. Domain adaptation-ийн үндсэн зорилго нь энэ “domain shift” асуудлыг багасгах явдал юм.

•	Embedding / Representation Learning арга
Domain adaptable загваруудын амжилтын гол түлхүүр нь representation learning, өөрөөр хэлбэл домэйнээс хамаарал багатай текстийн дүрслэл сурах явдал юм. Үүний тулд дараах аргуудыг өргөн ашигладаг.
Уламжлалт түвшинд Word2Vec, GloVe зэрэг embedding-үүдийг ашиглаж, эх домэйн болон зорилтот домэйн дээр хамтад нь сургах аргыг хэрэглэдэг. Ингэснээр ижил утгатай боловч өөр домэйнд өөрөөр хэрэглэгддэг үгсийг ойролцоо вектор орон зайд байрлуулах боломж бүрддэг.
Орчин үеийн deep learning суурьтай аргачлалд BERT, RoBERTa, DistilBERT зэрэг pretrained language model-уудыг ашигладаг. Эдгээр загварууд нь их хэмжээний олон домэйны текст дээр урьдчилан сурсан тул домэйн хоорондын ялгааг тодорхой хэмжээнд “саармагжуулсан” representation үүсгэх чадвартай. Domain adaptation хийхдээ эдгээр загваруудыг source domain дээр fine-tune хийгээд, target domain дээр бага хэмжээний өгөгдлөөр дахин тохируулдаг.

•	Hyperparameter-ууд
Domain adaptable sentiment analysis загварт дараах hyperparameter-ууд чухал нөлөө үзүүлдэг.
Learning rate нь pretrained загваруудын хувьд маш бага (ихэвчлэн 1e-5 – 5e-5) байх шаардлагатай бөгөөд энэ нь өмнө сурсан хэлний мэдлэгийг эвдэхгүйгээр домэйнд тохируулах боломж олгодог. Batch size нь GPU санах ой болон өгөгдлийн хэмжээнээс хамаарч 16–32 орчим байдаг. Epoch-ийн тоо ихэвчлэн бага (2–5 epoch) байдаг бөгөөд хэт олон epoch нь source domain-д хэт таарах (overfitting) эрсдэлтэй.
Хэрэв adversarial domain adaptation ашиглаж байгаа бол domain classifier-ийн loss weight, gradient reversal layer-ийн коэффициент зэрэг нэмэлт hyperparameter-ууд орж ирдэг бөгөөд эдгээр нь домэйн ялгааг “мартагнуулах” түвшинг зохицуулдаг.
Ашигласан Deep Learning болон Machine Learning аргууд
Machine Learning талаас:
o	Logistic Regression
o	Support Vector Machine (SVM)
o	Naive Bayes
эдгээрийг ихэвчлэн baseline загвар болгон ашигладаг. Эдгээр загварууд нь домэйн солигдоход гүйцэтгэл огцом унадаг нь domain adaptation-ийн шаардлагыг тодорхой харуулдаг.
Deep Learning талаас:
o	CNN (text convolution)
o	LSTM / Bi-LSTM
o	Transformer-based models (BERT, RoBERTa)
зэрэг загваруудыг ашигладаг. Ялангуяа Transformer суурьтай загварууд нь домэйн хоорондын ялгааг илүү сайн даван туулах чадвартай байдаг.

Ашиглагдсан Computer Science аргууд
Энэхүү төрлийн загварууд дараах компьютер шинжлэх ухааны үндсэн ойлголтуудад тулгуурладаг.
o	Domain adaptation theory
o	Representation learning
o	Optimization (gradient descent, backpropagation)
o	Adversarial learning (GAN-тэй төстэй зарчим)
o	Regularization (dropout, weight decay)
o	Transfer learning
Ялангуяа adversarial learning ашигласан domain adaptation загварууд нь source болон target домэйны ялгааг ангилагч загвараар илрүүлж, үндсэн sentiment classifier-ийг энэ ялгаанд “мэдрэмтгий биш” болгох зарчмаар ажилладаг.

•	Үр дүн
Судалгааны үр дүнгээс харахад domain adaptation ашигласан загварууд нь target domain дээрх гүйцэтгэлийг энгийн source-only загваруудтай харьцуулахад мэдэгдэхүйц сайжруулдаг. Ялангуяа training өгөгдөл багатай target domain-д domain adaptable загварууд илүү давуу тал үзүүлдэг. Гэхдээ source болон target домэйн хоорондын ялгаа хэт их тохиолдолд (жишээ нь кино шүүмж → твиттер пост) гүйцэтгэлийн сайжрал хязгаарлагдмал байж болдог.

•	Ашигласан үнэлгээний арга
Загварын гүйцэтгэлийг дараах үнэлгээний аргуудаар хэмждэг.
o	Accuracy – нийт зөв ангилалтын хувь
o	Precision, Recall – эерэг болон сөрөг ангиллын чанарыг тусад нь үнэлэх
o	F1-score – precision болон recall-ийн тэнцвэрийг харуулна
o	Cross-domain evaluation – source domain дээр сургаад target domain дээр тестлэх
Эдгээр үнэлгээг хамтад нь ашигласнаар загвар зөвхөн source domain-д сайн ажиллаж байна уу, эсвэл үнэхээр domain adaptable чадвартай юу гэдгийг бодитоор дүгнэх боломж бүрддэг.

Нэг өгүүлбэрээр дүгнэвэл
Domain adaptable sentiment analysis загвар нь “үг мэддэг” байхаас гадна, орчны өөрчлөлтийг ойлгодог загвар бүтээх оролдлого бөгөөд энэ нь бодит хэрэглээнд ойр, судалгааны үнэ цэн өндөртэй чиглэл юм.
### 2.2 Aspect term extraction for sentiment analysis in large movie reviews using Gini Index feature selection method and SVM classifier-
•	Товч танилцуулга
Aspect term extraction (ATE) нь sentiment analysis-ийн дэд асуудал бөгөөд зөвхөн тухайн текст эерэг эсвэл сөрөг эсэхийг тодорхойлохоос гадна ямар шинж чанар (aspect)–д хандаж ийм үнэлгээ өгч байгааг илрүүлэх зорилготой. Том хэмжээний кино шүүмжүүдийн хувьд нэг шүүмж дотор олон aspect (жишээ нь жүжиглэлт, зохиол, зураг авалт, хөгжим) зэрэгцэн орших тул уламжлалт sentiment analysis хангалтгүй болдог. Ийм нөхцөлд aspect term extraction нь илүү нарийн, тайлбарлах боломжтой дүн шинжилгээ хийх суурь болдог.
Энэхүү аргачлалд Gini Index-ийг feature selection хийхэд ашиглаж, сонгогдсон шинжүүд дээр Support Vector Machine (SVM) ангилагч ашиглан aspect term-үүдийг илрүүлэхэд чиглэсэн байна.

•	Embedding / Representation Learning арга
Энэхүү судалгаанд representation learning-ийг гүн сургалтын embedding-ээр бус, feature engineering-д суурилсан векторчлолын аргаар гүйцэтгэдэг нь онцлог юм. Үгс болон n-gram-уудыг Bag-of-Words эсвэл TF-IDF аргаар тоон хэлбэрт шилжүүлж, тухайн үг aspect term байх магадлалыг илэрхийлэх олон шинжүүдийг үүсгэдэг.
Gini Index нь эдгээр шинжүүдийн дундаас ангилалд хамгийн их ялгаатай мэдээлэл өгдөг feature-үүдийг сонгох шалгуур болж өгдөг. Үүний үр дүнд өндөр хэмжээст, шуугиантай feature орон зайг багасгаж, aspect term илрүүлэхэд илүү чухал үгсийн дүрслэлийг үлдээдэг. Энэ арга нь deep embedding ашиглаагүй ч домэйнд хамааралтай үгсийг үр дүнтэй ялгаж чаддаг.

•	Hyperparameter-ууд
SVM ангилагч ашигласан тул дараах hyperparameter-ууд чухал үүрэгтэй.
Penalty parameter (C) нь ангиллын алдаа болон margin-ийн тэнцвэрийг зохицуулдаг бөгөөд хэт их C нь overfitting, хэт бага C нь underfitting үүсгэх эрсдэлтэй. Kernel-ийн сонголт (ихэвчлэн linear kernel) нь өндөр хэмжээст текстийн өгөгдөлд тохиромжтой байдаг. Мөн feature selection шатанд Gini Index-ийн threshold буюу хэдэн feature үлдээх вэ гэдэг параметр нь гүйцэтгэлд шууд нөлөөлдөг.

•	Ашигласан Deep Learning болон Machine Learning аргууд
Энэхүү судалгаа нь deep learning-д бус, классик машин сургалтын арга-д төвлөрсөн нь онцлог.
Machine Learning:
o	Support Vector Machine (SVM) – үндсэн ангилагч
o	Decision Tree – Gini Index тооцоолох суурь
o	Logistic Regression – зарим судалгаанд baseline байдлаар
Deep Learning:
o	Ашиглаагүй (энэ нь судалгааны нэг онцлог бөгөөд feature selection-ийн үр нөлөөг харуулахад чиглэсэн)

•	Ашиглагдсан Computer Science аргууд
Уг аргачлал дараах компьютер шинжлэх ухааны үндсэн ойлголтууд дээр суурилдаг.
o	Feature selection
o	Information theory (Gini Index)
o	Text preprocessing (tokenization, POS tagging)
o	Sequence labeling асуудлын хялбаршуулсан хэлбэр
o	Convex optimization (SVM)
o	High-dimensional data processing
Ялангуяа Gini Index ашигласнаар aspect term болон энгийн үгсийн хоорондын мэдээллийн ялгааг статистик үндэслэлтэйгээр тодорхойлж чаддаг.


•	Үр дүн
Туршилтын үр дүнгээс харахад Gini Index ашигласан feature selection нь aspect term extraction-ийн нарийвчлалыг сайжруулж, SVM ангилагчийн гүйцэтгэлийг уламжлалт бүх feature ашигласан хувилбартай харьцуулахад илүү тогтвортой болгосон. Том хэмжээний кино шүүмжүүдийн хувьд feature-ийн тоо ихсэх тусам шуугиан нэмэгддэг тул энэхүү арга нь computational cost болон гүйцэтгэлийн хооронд сайн тэнцвэр олж чадсан.

•	Ашигласан үнэлгээний арга
Aspect term extraction нь ангиллын бодлогоос илүү нарийн тул дараах үнэлгээний аргуудыг ашигладаг.
o	Precision – илрүүлсэн aspect term-үүдийн зөв байдал
o	Recall – бодит aspect term-үүдээс хэдийг илрүүлсэн эсэх
o	F1-score – precision болон recall-ийн тэнцвэр
o	Token-level evaluation – үг бүрийн түвшинд үнэлэх
Эдгээр үзүүлэлтүүдийг ашигласнаар загвар зөвхөн олон aspect олж байна уу, эсвэл үнэхээр зөв aspect-уудыг ялгаж чадсан эсэхийг бодитоор үнэлэх боломж бүрддэг.

Дүгнэлт
Gini Index дээр суурилсан feature selection болон SVM ангилагчийг хослуулсан энэхүү арга нь deep learning ашиглалгүйгээр, том хэмжээний кино шүүмжүүдээс aspect term-үүдийг үр дүнтэй илрүүлэх боломжтойг харуулдаг. Энэ нь тооцооллын нөөц хязгаарлагдмал орчинд, эсвэл тайлбарлах боломж өндөртэй загвар шаардлагатай нөхцөлд практик ач холбогдолтой аргачлал гэж дүгнэж байна.

### 2.3 An intelligent sentiment prediction approach in social networks based on batch and streaming big data analytics using deep learning
•	Товч танилцуулга
Энэхүү судалгааны ажлын үндсэн зорилго нь сошиал сүлжээнд үүсэж буй их хэмжээний текст өгөгдлөөс хэрэглэгчдийн сэтгэл хөдлөлийг бодит цагийн (streaming) болон түүхэн (batch) горимд үр дүнтэй таамаглах ухаалаг систем боловсруулах явдал юм. Сошиал сүлжээний өгөгдөл нь өндөр хурдтай, бүтэцгүй, дуу чимээ ихтэй байдаг тул уламжлалт sentiment analysis аргууд бодит цагийн орчинд хангалттай гүйцэтгэл үзүүлдэггүй. Иймээс уг судалгаанд big data analytics болон deep learning-ийг хослуулсан гибрид архитектур санал болгосон.
Энэхүү хандлага нь офлайн (batch) өгөгдлөөс урт хугацааны хандлагыг суралцах, онлайн (streaming) өгөгдлөөс бодит цагийн сэтгэл хөдлөлийн өөрчлөлтийг илрүүлэх боломжийг нэг системд нэгтгэдгээрээ онцлог юм.

•	Embedding / Representation Learning арга
Representation learning нь уг системийн гол бүрэлдэхүүн хэсэг бөгөөд сошиал сүлжээний богино, албан бус текстийг үр дүнтэй дүрслэхэд чиглэсэн. Уламжлалт TF-IDF аргыг baseline байдлаар ашиглах боломжтой боловч deep learning-д суурилсан embedding-үүд илүү үр дүн үзүүлдэг.
Үүнд Word2Vec, GloVe зэрэг статик embedding-үүдийг batch өгөгдөл дээр урьдчилан сургаж, streaming өгөгдөлд дахин ашигладаг. Мөн LSTM болон CNN суурьтай загваруудын дотор embedding layer ашиглан өгөгдлөөс шууд dense representation сурах аргыг хэрэглэдэг. Орчин үеийн хувилбаруудад contextual embedding (жишээ нь BERT-ийн хөнгөн хувилбарууд) ашигласнаар сошиал сүлжээний богино өгүүлбэрийн утгыг илүү сайн барьж чаддаг.

•	Hyperparameter-ууд
Deep learning суурьтай sentiment prediction загварт дараах hyperparameter-ууд чухал нөлөөтэй.
Learning rate нь streaming орчинд тогтвортой байдал хангах үүднээс бага утгатай (1e-4 – 1e-5) сонгогддог. Batch size нь batch analytics-д харьцангуй том (64–256), харин streaming analytics-д micro-batch хэлбэрээр (жишээ нь 16–32) ашиглагддаг. LSTM-ийн hidden unit-ийн тоо, CNN-ийн filter-ийн хэмжээ, dropout rate зэрэг нь overfitting-ээс сэргийлэхэд чухал үүрэгтэй. Мөн streaming системд sliding window size зэрэг параметрүүд бодит цагийн хариу үйлдлийн хурдыг тодорхойлдог.

•	Ашигласан Deep Learning болон Machine Learning аргууд
Machine Learning талаас:
o	Naive Bayes
o	Logistic Regression
o	Support Vector Machine
зэргийг baseline загвар болгон ашиглаж, deep learning-ийн давуу талыг харьцуулж үздэг.
Deep Learning талаас:
o	CNN – орон нутгийн текстийн хэв маяг илрүүлэх
o	LSTM / Bi-LSTM – цаг хугацааны дараалал, контекст ойлгох
o	Hybrid CNN-LSTM – богино ба урт хугацааны мэдээллийг хослуулах
o	Зарим хувилбарт GRU эсвэл attention механизм
Эдгээр загваруудыг batch болон streaming орчинд тохируулан ашигладаг.

•	Ашиглагдсан Computer Science аргууд
Энэхүү систем нь дараах компьютер шинжлэх ухааны үндсэн ойлголтуудад тулгуурладаг.
o	Big Data Analytics
o	Batch Processing (жишээ нь Hadoop, Spark)
o	Stream Processing (жишээ нь Spark Streaming, Kafka)
o	Distributed Computing
o	Deep Learning Optimization (SGD, Adam)
o	Concept drift detection
o	Scalable system architecture
Ялангуяа batch ба streaming аналитикийг нэгтгэснээр систем нь урт хугацааны хэв маяг болон бодит цагийн өөрчлөлтийг зэрэг барих боломжтой болдог.

•	Үр дүн
Туршилтын үр дүнгээс харахад deep learning-д суурилсан batch + streaming гибрид загварууд нь уламжлалт machine learning загваруудтай харьцуулахад илүү өндөр нарийвчлал, тогтвортой байдал үзүүлсэн. Ялангуяа бодит цагийн өгөгдөлд sentiment-ийн огцом өөрчлөлтийг илрүүлэх чадвар сайжирсан нь ажиглагдсан. Гэхдээ системийн нарийн төвөгтэй байдал нэмэгдэж, тооцооллын зардал өсөх сул талтай.

•	Ашигласан үнэлгээний арга
Системийн гүйцэтгэлийг дараах аргуудаар үнэлдэг.
o	Accuracy – нийт зөв таамаглалын хувь
o	Precision, Recall – эерэг ба сөрөг ангиллын чанарыг тусад нь үнэлэх
o	F1-score – ангиллын тэнцвэрийг харуулах
o	Latency – бодит цагийн хариу үйлдлийн хурд
o	Throughput – нэгж хугацаанд боловсруулсан өгөгдлийн хэмжээ
Эдгээр үзүүлэлтүүд нь загвар зөвхөн “зөв таамаглаж байна уу” гэдгээс гадна, бодит системд ашиглахад тохиромжтой эсэхийг дүгнэхэд чухал.

Дүгнэлт
Batch болон streaming big data analytics-ийг deep learning-тэй хослуулсан энэхүү intelligent sentiment prediction approach нь сошиал сүлжээний динамик, их хэмжээний өгөгдөлд бодит цагийн сэтгэл хөдлөлийн шинжилгээ хийх боломжтойг харуулж байна. Энэ нь судалгааны хувьд ч, бодит хэрэглээний хувьд ч өндөр ач холбогдолтой шийдэл гэж үзэж байна.
### 2.4 Classification of tweets data based on polarity using improved RBF kernel of SVM
•	Товч танилцуулга
Энэхүү судалгааны ажлын зорилго нь Twitter зэрэг сошиал сүлжээнд нийтлэгдсэн богино текстүүдийн сэтгэл хөдлөлийн чиглэлийг (эерэг, сөрөг, зарим тохиолдолд төвийг сахисан) тодорхойлох явдал юм. Tweet өгөгдөл нь богино хэмжээтэй, албан бус хэллэгтэй, hashtag, emoji, товчилсон үг их агуулдаг тул уламжлалт текст ангиллын аргуудад хүндрэлтэй байдаг. Иймээс уг судалгаанд Support Vector Machine (SVM)-ийг сайжруулсан Radial Basis Function (RBF) kernel-тэйгээр ашиглаж, өгөгдлийн шугаман бус тархалтыг илүү сайн загварчлахыг зорьсон.

•	Embedding / Representation Learning арга
Энэхүү аргачлалд representation learning-ийг deep embedding-ээр бус, статистик векторчлолын аргаар гүйцэтгэдэг. Tweet өгөгдлийг урьдчилан боловсруулахдаа URL, mention, hashtag, emoji-г зохих байдлаар цэвэрлэж эсвэл тусгай token болгон хувиргана. Үүний дараа Bag-of-Words эсвэл TF-IDF аргаар текстийг өндөр хэмжээст вектор хэлбэрт оруулдаг.
Improved RBF kernel нь эдгээр өндөр хэмжээст, шугаман бус тархалттай өгөгдлийг implicit байдлаар илүү баялаг feature орон зайд дүрслэх боломж олгодог. Өөрөөр хэлбэл, explicit embedding сурахгүй ч kernel trick ашиглан илүү илэрхий representation бий болгодог.

•	Hyperparameter-ууд
SVM болон RBF kernel ашигласан загварт дараах hyperparameter-ууд чухал үүрэгтэй.
Penalty parameter (C) нь алдаа гаргах болон margin өргөсгөх хоорондын тэнцвэрийг зохицуулдаг. Gamma параметр нь RBF kernel-ийн нөлөөллийн радиусыг тодорхойлж, нэг өгөгдлийн цэг ойр орчиндоо хэр хүчтэй нөлөөлөхийг илэрхийлнэ. Improved RBF kernel-ийн хувьд gamma-г динамик байдлаар тохируулах, эсвэл өгөгдлийн тархалтаас хамааруулан шинэчилсэн функц ашиглах замаар стандарт RBF kernel-ээс илүү уян хатан болгосон байдаг. Эдгээр параметрүүдийг grid search эсвэл cross-validation ашиглан сонгодог.
•	Ашигласан Deep Learning болон Machine Learning аргууд
Machine Learning талаас:
o	Support Vector Machine (SVM) with improved RBF kernel – үндсэн ангилагч
o	Naive Bayes, Logistic Regression – baseline харьцуулалт
Deep Learning:
o	Ашиглаагүй (энэ нь богино текст, харьцангуй бага өгөгдөлд kernel-based арга илүү үр дүнтэй байж болохыг харуулах зорилготой)

•	Ашиглагдсан Computer Science аргууд
Энэхүү судалгаанд дараах компьютер шинжлэх ухааны үндсэн ойлголтууд ашиглагдсан.
o	Kernel methods
o	Convex optimization
o	High-dimensional data processing
o	Text preprocessing (tokenization, normalization)
o	Feature scaling
o	Model selection and validation
Ялангуяа improved RBF kernel ашигласнаар tweet өгөгдлийн шугаман бус бүтэц, олон янзын хэллэгийг илүү сайн загварчлах боломж бүрдсэн.

•	Үр дүн
Туршилтын үр дүнгээс харахад сайжруулсан RBF kernel-тэй SVM нь стандарт linear kernel болон уламжлалт RBF kernel-тэй SVM-ээс polarity classification-ийн хувьд илүү өндөр гүйцэтгэл үзүүлсэн. Ялангуяа эерэг болон сөрөг ангиллын зааг бүдэг үед improved kernel нь илүү тогтвортой шийдвэр гаргаж чадсан. Энэ нь tweet өгөгдлийн шугаман бус шинж чанарыг илүү сайн барьж чадсаны илрэл гэж үздэг.
______________________________________
•	Ашигласан үнэлгээний арга
Загварын гүйцэтгэлийг дараах үзүүлэлтүүдээр үнэлсэн.
o	Accuracy – нийт зөв ангилалтын хувь
o	Precision – эерэг болон сөрөг ангиллын нарийвчлал
o	Recall – бодит polarity-ийг олж чадсан хувь
o	F1-score – precision болон recall-ийн тэнцвэр
o	Confusion matrix – ангиллын алдааны бүтэц
Cross-validation ашигласнаар загварын ерөнхий чадварыг илүү найдвартай үнэлсэн.
______________________________________
Дүгнэлт
Improved RBF kernel-тэй SVM-д суурилсан энэхүү арга нь tweet өгөгдлийн богино, шугаман бус бүтэцтэй текстийг үр дүнтэй ангилах боломжтойг харуулж байна. Deep learning ашиглаагүй ч kernel-based machine learning арга нь тодорхой нөхцөлд илүү тайлбарлах боломжтой, тогтвортой шийдэл болж чаддагийг энэхүү судалгаа баталж байна.
### 2.5 Distributed representations of sentences and documents
•	Товч танилцуулга
Distributed representations of sentences and documents гэдэг нь өгүүлбэр болон баримт бичгийг тогтмол урттай, нягт (dense) вектор хэлбэрээр дүрслэх аргачлал юм. Энэхүү санаа нь үгийг тус тусад нь бус, бүхэл өгүүлбэр эсвэл баримт бичгийг утгын орон зайд байрлуулах зорилготой. Энэ аргыг анх Word2Vec-ийн логик дээр суурилан өргөжүүлж, өгүүлбэр болон баримт бичгийн түвшинд хэрэгжүүлсэн нь Doc2Vec загвар юм.
Уламжлалт Bag-of-Words эсвэл TF-IDF аргууд нь үгсийн дараалал, утгын холбоог бүрэн хадгалж чаддаггүй бол distributed representation нь текстийн ерөнхий сэдэв, утга, хэв маягийг нэг вектороор илэрхийлэх боломж олгодог.



•	Embedding / Representation Learning арга
Энэхүү судалгаанд ашиглагддаг гол representation learning арга нь Doc2Vec бөгөөд энэ нь Word2Vec-ийн хоёр үндсэн архитектур дээр суурилдаг.
Distributed Memory (DM) загвар нь баримт бичгийн векторыг контекст үгсийн хамт ашиглан дараагийн үгийг таамаглах замаар сургагддаг. Ингэснээр document vector нь тухайн баримт бичгийн утга, сэдвийг хадгалсан “санах ой” болж өгдөг. Distributed Bag of Words (DBOW) загвар нь баримт бичгийн векторыг ашиглан тухайн баримтад багтах үгсийг таамаглах бөгөөд илүү хялбар бүтэцтэй, тооцооллын хувьд хөнгөн байдаг.
Эдгээр аргууд нь supervised шошго шаардалгүйгээр, их хэмжээний текст өгөгдлөөс өөрөө утгын representation сурах боломжтой гэдгээрээ онцлог.

•	Hyperparameter-ууд
Doc2Vec загварт дараах hyperparameter-ууд чухал нөлөө үзүүлдэг.
Vector size нь баримт бичгийн векторын хэмжээг тодорхойлж, хэт бага байвал утга алдагдах, хэт их байвал overfitting болон тооцооллын зардал нэмэгддэг. Window size нь контекстийн өргөнийг тодорхойлж, өгүүлбэрийн орон нутгийн болон глобал мэдээллийн аль алиныг сурахад нөлөөлнө. Epoch-ийн тоо нь сургалтын давтамжийг илэрхийлдэг бөгөөд хангалтгүй үед embedding чанар муудна. Мөн negative sampling-ийн тоо, learning rate зэрэг параметрүүд representation-ийн тогтвортой байдалд чухал.

•	Ашигласан Deep Learning болон Machine Learning аргууд
Machine Learning талаас:
o	Doc2Vec – үндсэн representation learning загвар
o	Logistic Regression, SVM – document vector дээр суурилсан ангилалд
Deep Learning:
o	Deep neural network ашиглаагүй
o	Гэсэн ч нейрон сүлжээний зарчимд суурилсан shallow model гэж үздэг
Энэ нь Doc2Vec нь deep architecture ашиглахгүй ч embedding сурах чадвараараа NLP-д чухал байр суурь эзэлдэгийг харуулдаг.

•	Ашиглагдсан Computer Science аргууд
Энэхүү аргачлал дараах компьютер шинжлэх ухааны үндсэн ойлголтууд дээр суурилдаг.
o	Distributional semantics
o	Representation learning
o	Stochastic gradient descent
o	Negative sampling
o	Unsupervised learning
o	Vector space models
Ялангуяа “утга нь хамт гарч ирдэг үгсийн орчинд оршдог” гэсэн distributional hypothesis нь энэхүү аргын философийн суурь юм.

•	Үр дүн
Туршилтын үр дүнгээс харахад Doc2Vec-ээр үүсгэсэн document vector-ууд нь ангилал, кластерчлал, ижил төстэй баримт хайлт зэрэг олон NLP даалгаварт TF-IDF-ээс илүү сайн гүйцэтгэл үзүүлсэн. Ялангуяа баримт бичгийн урт нэмэгдэх тусам distributed representation нь илүү давуу талтай болж байгаа нь ажиглагдсан. Гэхдээ маш богино текст (жишээ нь tweet) дээр гүйцэтгэл хязгаарлагдмал байж болдог.

•	Ашигласан үнэлгээний арга
Distributed representation-ийн чанарыг дараах аргуудаар үнэлдэг.
o	Classification accuracy – document vector ашигласан ангиллын гүйцэтгэл
o	Similarity tasks – cosine similarity ашиглан ижил төстэй баримт илрүүлэх
o	Clustering performance – K-means зэрэг аргаар бүлэглэх
o	Downstream task evaluation – embedding-ийг бодит даалгаварт ашиглаж шалгах
Эдгээр нь embedding өөрөө “сайн” эсэхийг бус, ашиглахад хэр үр дүнтэй вэ гэдгийг хэмждэг.

Дүгнэлт
Distributed representations of sentences and documents нь текстийг бүхэлд нь утгын орон зайд байрлуулах боломж олгосон NLP-ийн чухал ахиц юм. Doc2Vec нь deep learning-ээс өмнөх үеийн загвар боловч representation learning-ийн суурь ойлголтыг тодорхойлж, өнөөгийн Transformer суурьтай embedding-үүдийн онолын эхлэл болсон гэж дүгнэж болно.

### 2.6 Baselines and bigrams: Simple, good sentiment and topic classification
Товч танилцуулга
“Baselines and Bigrams” өгүүлэл нь sentiment болон topic classification-д заавал нарийн, төвөгтэй загвар ашиглах шаардлагагүй, зөв feature engineering болон тохирсон ангилагч ашиглавал маш сайн үр дүн гаргаж болдгийг харуулсан судалгаа юм. Тус ажил нь ялангуяа unigram + bigram feature-ийг linear classifier-тай хослуулахад тухайн үеийн олон илүү төвөгтэй загваруудтай өрсөлдөхүйц гүйцэтгэл гарсныг харуулснаараа онцлог.
Энэхүү судалгаа нь “baseline загвар бол заавал сул” гэсэн ойлголтыг эвдэж, бодит судалгаанд зөв baseline сонголт ямар чухал вэ гэдгийг тод харуулсан.
______________________________________
Embedding / Representation Learning арга
Энэхүү аргачлалд deep embedding ашиглаагүй. Representation learning нь explicit feature engineering дээр суурилсан. Текстийг unigram болон bigram болгон задлаж, эдгээрийг Bag-of-Words хэлбэрээр векторчилдог.
Онцлох нэг шийдэл нь binarized features буюу тухайн unigram эсвэл bigram тухайн баримтад орсон эсэхийг 0/1 утгаар тэмдэглэсэн явдал юм. Энэ нь TF эсвэл TF-IDF-ээс ялгаатайгаар үгийн давтамжийн хэт нөлөөг багасгаж, sentiment болон topic-д илүү чухал хэв маягийг илрүүлэхэд тусалсан.
______________________________________
Hyperparameter-ууд
Linear classifier ашигласан тул hyperparameter-ууд харьцангуй цөөн боловч чDistributed Representations of Sentences and Documentsухал.
Regularization parameter (λ эсвэл C) нь overfitting-ээс сэргийлэх үндсэн механизм болдог. Feature set-ийн хэмжээ (unigram only эсвэл unigram + bigram) нь гүйцэтгэлд шууд нөлөөлдөг гол параметр юм. Мөн binarization ашиглах эсэх нь загварын тогтвортой байдалд ихээхэн нөлөө үзүүлсэн.
______________________________________
Ашигласан Deep Learning болон Machine Learning аргууд
Machine Learning:
o	Linear SVM
o	Logistic Regression (Maximum Entropy)
Deep Learning:
o	Ашиглаагүй
Энэ нь судалгааны гол санаатай нийцэж, энгийн шугаман загвар зөв feature-үүдтэй хослоход маш хүчтэй болохыг харуулсан.
______________________________________
•	Ашиглагдсан Computer Science аргууд
Энэхүү судалгаа дараах компьютер шинжлэх ухааны ойлголтууд дээр суурилсан.
o	Feature engineering
o	N-gram language modeling
o	Linear classification
o	Regularization
o	High-dimensional sparse vectors
o	Empirical evaluation of baselines
Ялангуяа bigram ашигласнаар “not good”, “very bad” зэрэг sentiment-ийн хувьд чухал илэрхийлэл unigram-аар баригдахгүй сул талыг нөхсөн.
______________________________________
•	Үр дүн
Туршилтын үр дүнгээс харахад unigram + bigram, binarized feature ашигласан linear classifier нь sentiment болон topic classification дээр тухайн үеийн state-of-the-art-д ойртсон, зарим тохиолдолд давсан гүйцэтгэл үзүүлсэн. Энэ нь feature сонголт буруу бол илүү нарийн загвар ч муу ажиллаж болдгийг харуулсан бодит жишээ болсон.
______________________________________
•	Ашигласан үнэлгээний арга
Загварын гүйцэтгэлийг дараах аргуудаар үнэлсэн.
o	Accuracy – нийт зөв ангилалтын хувь
o	Error rate – буруу ангилалтын түвшин
o	Cross-validation – загварын ерөнхий чадварыг шалгах
o	Dataset-wise comparison – олон датасет дээр харьцуулалт хийх
Эдгээр үнэлгээ нь зөвхөн нэг даалгаварт бус, өөр өөр нөхцөлд загвар хэр тогтвортой ажиллаж байгааг харуулсан.
______________________________________
Дүгнэлт
“Baselines and Bigrams” судалгаа нь NLP-д энгийн шийдэл буруу гэсэн үг биш гэдгийг нотолсон ажил юм. Энэ нь оюутны судалгаанд baseline-ийг үл тоомсорлох бус, харин ухаалгаар сонгож, гүн загвартай шударгаар харьцуулах ёстой гэсэн чухал сургамж өгдөг.

### 2.7 Analysis of IMDB reviews for movies and television series using SAS® Enterprise Miner™ and SAS® Sentiment Analysis Studio
Товч танилцуулга
Энэхүү судалгааны ажил нь IMDB вэб сайтад байршуулсан кино болон телевизийн цувралын хэрэглэгчдийн шүүмжийг SAS® Enterprise Miner™ болон SAS® Sentiment Analysis Studio ашиглан дүн шинжилгээ хийхэд чиглэсэн. Судалгааны үндсэн зорилго нь бодит хэрэглэгчдийн бичсэн их хэмжээний текст өгөгдлөөс сэтгэл хөдлөлийн чиг хандлагыг илрүүлэх, мөн кино ба телевизийн цувралын шүүмжүүдийн хоорондын ялгааг аналитик аргаар тодорхойлох явдал юм.
Энэхүү ажил нь академик судалгаанаас гадна аж үйлдвэрийн түвшний аналитик платформ ашиглан sentiment analysis хэрхэн хэрэгжиж байгааг харуулснаараа онцлог.

•	Embedding / Representation Learning арга
SAS орчинд representation learning нь гүн сургалтын embedding-ээс илүүтэйгээр linguistic rule-based болон статистик шинжид суурилсан дүрслэл дээр тулгуурладаг. SAS Sentiment Analysis Studio нь урьдчилан тодорхойлсон sentiment dictionary, дүрмийн загвар (grammar rules), мөн token-level оноолтыг ашиглан текстийн утгыг илэрхийлдэг.
Enterprise Miner орчинд текстийг term, phrase, part-of-speech, dependency зэрэг шинжүүдээр векторчилж, TF, TF-IDF, term frequency weight зэрэг уламжлалт representation-уудыг ашигладаг. Энэ нь embedding-ээс илүү тайлбарлах боломж өндөртэй дүрслэл үүсгэдэг.
______________________________________
•	Hyperparameter-ууд
SAS орчинд hyperparameter нь шууд кодоор бус, визуал тохиргоогоор удирдагддаг нь онцлог.
Enterprise Miner-д:
o	Feature selection threshold
o	Term weighting scheme (TF vs TF-IDF)
o	Minimum term frequency
o	Stopword list болон stemming тохиргоо
Sentiment Analysis Studio-д:
o	Sentiment score threshold
o	Rule priority
o	Polarity classification sensitivity
Эдгээр параметрүүдийг тохируулснаар загварын нарийвчлал болон тайлбарлах чадварыг удирддаг.
______________________________________
•	Ашигласан Deep Learning болон Machine Learning аргууд
Machine Learning:
o	Decision Tree
o	Logistic Regression
o	Naive Bayes
o	Support Vector Machine (Enterprise Miner дотор)
Deep Learning:
o	Шууд deep neural network ашиглаагүй
o	Харин rule-based sentiment engine ашигласан
Энэ нь судалгааны гол санаатай нийцэж, deep learning заавал шаардлагатай биш, харин domain-д тохирсон аналитик хэрэгсэл үр дүнтэй байж болохыг харуулсан.

•	Ашиглагдсан Computer Science аргууд
Энэхүү судалгаа дараах компьютер шинжлэх ухааны үндсэн ойлголтуудад тулгуурладаг.
o	Text mining
o	Natural Language Processing
o	Rule-based sentiment analysis
o	Statistical classification
o	Feature engineering
o	Data preprocessing and normalization
o	Visual analytics pipeline design
Ялангуяа SAS платформ нь алгоритмаас гадна workflow engineering буюу өгөгдлөөс үр дүн хүртэлх бүх процессыг системтэйгээр удирдах боломж олгодог.

•	Үр дүн
Судалгааны үр дүнгээс харахад кино болон телевизийн цувралын шүүмжүүдийн sentiment тархалт ялгаатай байгааг илрүүлсэн. Киноны шүүмжүүд илүү огцом (маш эерэг эсвэл маш сөрөг) хандлагатай байхад, телевизийн цувралуудын шүүмжүүд харьцангуй тогтвортой, олон улирлын явцад өөрчлөгдөх хандлагатай байсан. SAS Sentiment Analysis Studio ашигласнаар domain-д тохирсон sentiment илрүүлэлт илүү тайлбарлах боломжтой үр дүн өгсөн.
______________________________________
•	Ашигласан үнэлгээний арга
Системийн гүйцэтгэлийг дараах аргуудаар үнэлсэн.
o	Accuracy – нийт зөв ангилалтын хувь
o	Precision, Recall – эерэг болон сөрөг sentiment-ийн чанар
o	Confusion Matrix – ангиллын алдааны бүтэц
o	Rule-based validation – эксперт үнэлгээтэй харьцуулах
o	Comparative analysis – кино vs телевизийн цуврал
Эдгээр үнэлгээ нь загвар зөв таамаглаж байна уу гэдгээс гадна, аналитик үр дүн бизнес болон судалгааны тайлбарт ашиглагдах боломжтой эсэхийг харуулсан.
______________________________________
Дүгнэлт
IMDB шүүмжүүдийг SAS® Enterprise Miner™ болон SAS® Sentiment Analysis Studio ашиглан шинжилсэн энэхүү судалгаа нь sentiment analysis-ийг deep learning-гүйгээр, тайлбарлах боломж өндөртэй, үйлдвэрлэлийн түвшний аналитик орчинд амжилттай хэрэгжүүлж болдгийг харуулж байна. Энэ нь бодит байгууллагын аналитик шийдэлд чухал ач холбогдолтой хандлага юм.

### 2.8 SAS-based IMDB Analysis
Товч танилцуулга
SAS-based IMDB Analysis нь кино болон телевизийн цувралын хэрэглэгчдийн IMDB шүүмжийг аж үйлдвэрийн түвшний аналитик платформ болох SAS орчинд sentiment analysis хийж судалсан ажил юм. Судалгааны зорилго нь их хэмжээний бодит хэрэглэгчийн текст өгөгдлөөс сэтгэл хөдлөлийн чиг хандлагыг илрүүлэхдээ rule-based NLP болон классик машин сургалтын аргуудыг хэрхэн үр дүнтэй ашиглаж болохыг харуулахад оршино. Энэхүү ажил нь академик алгоритмаас гадна бодит бизнесийн аналитик pipeline-ийг онцолсноороо ялгардаг.

•	Embedding / Representation Learning арга
SAS орчинд embedding-ийг deep learning-ийн байдлаар сургах бус, тайлбарлах боломж өндөртэй текстийн дүрслэл ашигладаг. Үүнд:
o	Token, term, phrase-д суурилсан Bag-of-Words
o	TF, TF-IDF жинлэл
o	Part-of-Speech, dependency, phrase-level шинжүүд
o	Sentiment dictionary болон дүрмийн (grammar) дүрслэл
Эдгээр нь explicit (hand-crafted) representation тул аль үг, аль дүрэм сэтгэл хөдлөлийг тодорхойлсныг шууд тайлбарлах боломжтой.

•	Hyperparameter-ууд
SAS Enterprise Miner болон Sentiment Analysis Studio-д hyperparameter-уудыг кодоор бус визуал тохиргоо-гоор удирддаг. Гол параметрүүд:
o	Minimum / maximum term frequency
o	Term weighting scheme (TF vs TF-IDF)
o	Feature selection threshold
o	Stopword, stemming тохиргоо
o	Sentiment score threshold
o	Rule priority болон polarity sensitivity
Эдгээрийг тохируулах замаар нарийвчлал ба тайлбарлах чадварын тэнцвэр-ийг барьдаг.

•	Ашигласан Deep Learning болон Machine Learning аргууд
Machine Learning:
o	Logistic Regression
o	Decision Tree
o	Naive Bayes
o	Support Vector Machine (SAS дотор)
Deep Learning:
o	Ашиглаагүй
o	Харин rule-based sentiment engine ашигласан
Энэ нь deep learning заавал шаардлагатай биш, домэйнд тохирсон дүрэм + статистик загвар бодит хэрэглээнд үр дүнтэй байж болохыг харуулдаг.

•	Ашиглагдсан Computer Science аргууд
o	Text mining
o	Natural Language Processing
o	Rule-based sentiment analysis
o	Feature engineering
o	Statistical classification
o	Data preprocessing (normalization, tokenization)
o	Visual workflow design (pipeline-based analytics)
Ялангуяа workflow engineering буюу өгөгдлөөс тайлан хүртэлх процессыг системтэй удирдах нь SAS-ийн гол давуу тал.

•	Үр дүн
Судалгааны үр дүнгээс харахад кино болон телевизийн цувралын шүүмжүүдийн sentiment тархалт ялгаатай байсан. Киноны шүүмжүүд илүү туйлширсан (маш эерэг/маш сөрөг) байхад, телевизийн цувралуудын шүүмжүүд улирлын явцад аажмаар өөрчлөгдөх хандлагатай байв. Rule-based SAS sentiment engine нь domain-specific үг хэллэгийг сайн барьж, тогтвортой үр дүн үзүүлсэн.

•	Ашигласан үнэлгээний арга
o	Accuracy
o	Precision
o	Recall
o	F1-score
o	Confusion Matrix
o	Expert validation (rule-based үр дүнг гараар шалгах)
o	Comparative analysis (movie vs TV series)
Эдгээр нь загварын тоон гүйцэтгэл-ээс гадна бодит хэрэглээнд тайлбарлах боломж-ийг хамтад нь үнэлэхэд чиглэсэн.

Дүгнэлт
SAS-based IMDB Analysis нь sentiment analysis-ийг deep learning-гүйгээр, тайлбарлах боломж өндөртэй, үйлдвэрлэлийн түвшний аналитик орчинд амжилттай хэрэгжүүлж болдгийг харуулсан судалгаа юм. Энэ нь судалгааны ажлаас гадна бизнес, байгууллагын шийдвэр гаргалтад NLP-г бодитоор ашиглах боломжийг тодорхойлсон ач холбогдолтой.




### 2.9 ANN & RNN for Movie Reviews (Thesis, 2023)
Товч танилцуулга
Энэхүү судалгааны ажил нь IMDB вэб сайтад байршуулсан кино болон телевизийн цувралын хэрэглэгчдийн шүүмжийг SAS® Enterprise Miner™ болон SAS® Sentiment Analysis Studio ашиглан дүн шинжилгээ хийхэд чиглэсэн. Судалгааны үндсэн зорилго нь бодит хэрэглэгчдийн бичсэн их хэмжээний текст өгөгдлөөс сэтгэл хөдлөлийн чиг хандлагыг илрүүлэх, мөн кино ба телевизийн цувралын шүүмжүүдийн хоорондын ялгааг аналитик аргаар тодорхойлох явдал юм.
Энэхүү ажил нь академик судалгаанаас гадна аж үйлдвэрийн түвшний аналитик платформ ашиглан sentiment analysis хэрхэн хэрэгжиж байгааг харуулснаараа онцлог
•	Embedding / Representation Learning арга
SAS орчинд representation learning нь гүн сургалтын embedding-ээс илүүтэйгээр linguistic rule-based болон статистик шинжид суурилсан дүрслэл дээр тулгуурладаг. SAS Sentiment Analysis Studio нь урьдчилан тодорхойлсон sentiment dictionary, дүрмийн загвар (grammar rules), мөн token-level оноолтыг ашиглан текстийн утгыг илэрхийлдэг.
Enterprise Miner орчинд текстийг term, phrase, part-of-speech, dependency зэрэг шинжүүдээр векторчилж, TF, TF-IDF, term frequency weight зэрэг уламжлалт representation-уудыг ашигладаг. Энэ нь embedding-ээс илүү тайлбарлах боломж өндөртэй дүрслэл үүсгэдэг.
•	Hyperparameter-ууд
SAS орчинд hyperparameter нь шууд кодоор бус, визуал тохиргоогоор удирдагддаг нь онцлог.
Enterprise Miner-д:
o	Feature selection threshold
o	Term weighting scheme (TF vs TF-IDF)
o	Minimum term frequency
o	Stopword list болон stemming тохиргоо
Sentiment Analysis Studio-д:
o	Sentiment score threshold
o	Rule priority
o	Polarity classification sensitivity
Эдгээр параметрүүдийг тохируулснаар загварын нарийвчлал болон тайлбарлах чадварыг удирддаг.

•	Ашигласан Deep Learning болон Machine Learning аргууд
Machine Learning:
o	Decision Tree
o	Logistic Regression
o	Naive Bayes
o	Support Vector Machine (Enterprise Miner дотор)
Deep Learning:
o	Шууд deep neural network ашиглаагүй
o	Харин rule-based sentiment engine ашигласан
Энэ нь судалгааны гол санаатай нийцэж, deep learning заавал шаардлагатай биш, харин domain-д тохирсон аналитик хэрэгсэл үр дүнтэй байж болохыг харуулсан.

•	Ашиглагдсан Computer Science аргууд
Энэхүү судалгаа дараах компьютер шинжлэх ухааны үндсэн ойлголтуудад тулгуурладаг.
o	Text mining
o	Natural Language Processing
o	Rule-based sentiment analysis
o	Statistical classification
o	Feature engineering
o	Data preprocessing and normalization
o	Visual analytics pipeline design
Ялангуяа SAS платформ нь алгоритмаас гадна workflow engineering буюу өгөгдлөөс үр дүн хүртэлх бүх процессыг системтэйгээр удирдах боломж олгодог.

•	Үр дүн
Судалгааны үр дүнгээс харахад кино болон телевизийн цувралын шүүмжүүдийн sentiment тархалт ялгаатай байгааг илрүүлсэн. Киноны шүүмжүүд илүү огцом (маш эерэг эсвэл маш сөрөг) хандлагатай байхад, телевизийн цувралуудын шүүмжүүд харьцангуй тогтвортой, олон улирлын явцад өөрчлөгдөх хандлагатай байсан. SAS Sentiment Analysis Studio ашигласнаар domain-д тохирсон sentiment илрүүлэлт илүү тайлбарлах боломжтой үр дүн өгсөн.

•	Ашигласан үнэлгээний арга
Системийн гүйцэтгэлийг дараах аргуудаар үнэлсэн.
o	Accuracy – нийт зөв ангилалтын хувь
o	Precision, Recall – эерэг болон сөрөг sentiment-ийн чанар
o	Confusion Matrix – ангиллын алдааны бүтэц
o	Rule-based validation – эксперт үнэлгээтэй харьцуулах
o	Comparative analysis – кино vs телевизийн цуврал
Эдгээр үнэлгээ нь загвар зөв таамаглаж байна уу гэдгээс гадна, аналитик үр дүн бизнес болон судалгааны тайлбарт ашиглагдах боломжтой эсэхийг харуулсан.

Дүгнэлт
IMDB шүүмжүүдийг SAS® Enterprise Miner™ болон SAS® Sentiment Analysis Studio ашиглан шинжилсэн энэхүү судалгаа нь sentiment analysis-ийг deep learning-гүйгээр, тайлбарлах боломж өндөртэй, үйлдвэрлэлийн түвшний аналитик орчинд амжилттай хэрэгжүүлж болдгийг харуулж байна. Энэ нь бодит байгууллагын аналитик шийдэлд чухал ач холбогдолтой хандлага юм.
### 2.10 IMDB Movie Review Dataset
Товч танилцуулга
Энэхүү судалгааны ажлын зорилго нь кино шүүмжийн текст өгөгдөл дээр хиймэл нейрон сүлжээ (Artificial Neural Network, ANN) болон давталттай нейрон сүлжээ (Recurrent Neural Network, RNN)-ийн гүйцэтгэлийг харьцуулан судлахад оршино. Кино шүүмжүүд нь урт, өгүүлбэр хоорондын утга зүйн хамааралтай текст учраас энгийн feed-forward ANN болон дараалалд суурилсан RNN загваруудын ялгаа тод илэрдэг. Судалгаанд IMDB movie review dataset зэрэг өргөн хэрэглэгддэг өгөгдлийг ашиглан sentiment analysis даалгаврыг гүйцэтгэсэн.

•	Embedding / Representation Learning арга
Representation learning нь уг ажлын чухал бүрэлдэхүүн хэсэг юм. Эхний шатанд текст өгөгдлийг tokenization, stopword removal, padding зэрэг урьдчилсан боловсруулалтаар бэлтгэдэг. Үүний дараа үгсийг тоон орон зайд дүрслэхийн тулд embedding layer ашигладаг.
ANN загварын хувьд embedding layer-ийн гаралтыг flatten хийж, текстийг дарааллын мэдээлэлгүйгээр ерөнхий утгын вектор болгон ашигладаг. Харин RNN загварт embedding layer-ийн дараах векторуудыг дарааллын хэлбэрээр хадгалж, өгүүлбэр доторх үгсийн дараалал, контекстийг хадгалан боловсруулдаг. Зарим хувилбарт pretrained embedding (Word2Vec эсвэл GloVe) ашиглан загварын суралцах хурдыг нэмэгдүүлсэн.


•	Hyperparameter-ууд
ANN болон RNN загварт дараах hyperparameter-ууд чухал нөлөө үзүүлдэг.
Embedding dimension нь үгийн утгын багтаамжийг тодорхойлж, хэт бага үед мэдээлэл алдагдах, хэт их үед overfitting үүсэх эрсдэлтэй. Hidden layer-ийн neuron-уудын тоо нь загварын илэрхийлэх чадварыг тодорхойлдог. RNN загварт timestep буюу sequence length, hidden state-ийн хэмжээ онцгой чухал. Learning rate, batch size, epoch-ийн тоо, dropout rate зэрэг нь хоёр загварт хоёуланд нь тогтвортой сургалтыг хангахад ашиглагдсан.

•	Ашигласан Deep Learning болон Machine Learning аргууд
Deep Learning:
o	Feed-forward ANN
o	Simple RNN
o	Зарим хувилбарт LSTM эсвэл GRU (RNN-ийн сайжруулсан хэлбэр)
Machine Learning:
o	Logistic Regression
o	Naive Bayes
Эдгээр machine learning загваруудыг baseline болгон ашиглаж, ANN болон RNN-ийн давуу талыг харьцуулж үзсэн.

•	Ашиглагдсан Computer Science аргууд
Энэхүү судалгаа дараах компьютер шинжлэх ухааны үндсэн ойлголтуудад тулгуурласан.
o	Natural Language Processing
o	Representation learning
o	Sequence modeling
o	Backpropagation Through Time (BPTT)
o	Optimization algorithms (SGD, Adam)
o	Regularization (dropout, early stopping)
o	Model comparison and evaluation
Ялангуяа RNN загварт BPTT ашиглан дарааллын мэдээллийг суралцах нь ANN-ээс ялгарах гол онцлог болсон.

•	Үр дүн
Судалгааны үр дүнгээс харахад ANN загвар нь энгийн бүтэцтэй, сургалтын хурд сайн боловч өгүүлбэр доторх үгсийн дарааллыг харгалзан үздэггүй сул талтай байв. Харин RNN болон түүний сайжруулсан хувилбарууд нь урт шүүмжүүд дээр илүү өндөр нарийвчлал үзүүлж, sentiment-ийн өөрчлөлтийг илүү сайн барьж чадсан. Гэхдээ RNN загварууд нь тооцооллын зардал ихтэй, сургалт удаан үргэлжлэх сул талтай.

•	Ашигласан үнэлгээний арга
Загваруудын гүйцэтгэлийг дараах аргуудаар үнэлсэн.
o	Accuracy – нийт зөв ангилалтын хувь
o	Precision, Recall – эерэг болон сөрөг ангиллын чанар
o	F1-score – ангиллын тэнцвэр
o	Loss curve analysis – сургалтын явцын тогтвортой байдал
o	Confusion matrix – алдааны бүтэц
Эдгээр үзүүлэлтүүдийг ашиглан ANN болон RNN-ийн давуу ба сул талыг бодитоор харьцуулсан.

Дүгнэлт
ANN & RNN-ийг кино шүүмжийн sentiment analysis дээр харьцуулсан энэхүү судалгаа нь текстийн дарааллын мэдээлэл чухал үед RNN-ийн давуу тал тод илэрдэг болохыг харуулж байна. Гэсэн хэдий ч энгийн ANN загвар нь хурд, хэрэгжүүлэх хялбар байдлаараа зарим нөхцөлд практик ач холбогдолтой хэвээр байгааг уг ажил нотолж байна.
							

							
							





### 2.11 10 судалгааны ажлын харьцуулсан хүснэгт

| № | Судалгааны ажил | Өгөгдөл | Representation / Embedding | Ашигласан загвар | Онцлог арга | Үнэлгээ | Гол давуу тал |
|--:|-----------------|---------|----------------------------|------------------|------------|---------|---------------|
| 1 | Baselines and Bigrams (Wang & Manning) | IMDB, topic datasets | Unigram + Bigram (binary BoW) | Linear SVM, LR | Feature engineering | Accuracy | Энгийн baseline маш хүчтэй |
| 2 | Distributed Representations of Sentences and Documents | Олон текст корпус | Doc2Vec (DM, DBOW) | Doc2Vec + LR / SVM | Unsupervised embedding | Accuracy, Similarity | Баримт бичгийн утга хадгална |
| 3 | IMDB Movie Review Dataset | IMDB reviews | BoW, TF-IDF, Word2Vec | LR, SVM, CNN, RNN | Balanced dataset | Accuracy, F1 | NLP стандарт benchmark |
| 4 | Domain Adaptable Sentiment Model | Multi-domain reviews | Word2Vec, BERT | CNN, LSTM, Transformer | Domain adaptation | F1, Cross-domain | Domain shift багасгана |
| 5 | Aspect Term Extraction (Gini + SVM) | Movie reviews | TF-IDF + Feature selection | SVM | Gini Index | Precision, Recall, F1 | Aspect-level тайлбар |
| 6 | Intelligent Sentiment in Social Networks | Twitter, social data | Word2Vec, contextual embedding | CNN, LSTM, Hybrid | Batch + Streaming | Accuracy, Latency | Real-time анализ |
| 7 | Tweet Polarity using Improved RBF-SVM | Tweets | TF-IDF + Kernel trick | SVM (Improved RBF) | Kernel optimization | Accuracy, F1 | Богино текстэд сайн |
| 8 | SAS-based IMDB Analysis | IMDB reviews | Linguistic rules, TF-IDF | DT, NB, LR, SVM | Rule-based NLP | Accuracy | Industrial tool, explainable |
| 9 | ANN & RNN for Movie Reviews (2023) | IMDB reviews | Embedding layer, Word2Vec | ANN, RNN, LSTM | Sequence modeling | Accuracy, Loss | ANN vs RNN ялгаа |
|10 | IMDB Analysis (Student Thesis style) | IMDB reviews | BoW, TF-IDF, Embedding | LR, SVM, DL | Pipeline comparison | Accuracy, F1 | Боловсролын жишээ |


Эдгээр судалгааны ажлуудыг харьцуулан авч үзэхэд sentiment analysis-ийн үр дүн нь зөвхөн ашигласан загварын “гүнзгий байдал”-аас бус, өгөгдлийн шинж чанар, representation learning арга, feature сонголт-оос ихээхэн хамаарч байгааг ажиглаж болно. Зарим тохиолдолд unigram болон bigram дээр суурилсан энгийн baseline загварууд ч deep learning загваруудтай өрсөлдөхүйц гүйцэтгэл үзүүлж байгаа нь зөв representation болон тохирсон ангилагчийн ач холбогдлыг харуулж байна.
Нөгөө талаас, domain adaptation, streaming analytics зэрэг бодит хэрэглээний орчинд deep learning суурьтай загварууд илүү уян хатан, орчны өөрчлөлтөд дасан зохицох чадвартай болох нь харагдаж байна. Ялангуяа сошиал сүлжээний өгөгдөл шиг динамик, дуу чимээ ихтэй орчинд sequence болон contextual мэдээлэл ашиглах шаардлага улам нэмэгдэж байгаа нь RNN, Transformer зэрэг загваруудын ач холбогдлыг баталж байна.
Мөн судалгааны ажлуудын нэг чухал ялгаа нь тайлбарлах чадвар (interpretability) юм. Rule-based болон классик machine learning аргууд нь шийдвэр гаргалтыг тайлбарлахад илүү хялбар байдаг бол deep learning загварууд өндөр гүйцэтгэл үзүүлэхийн зэрэгцээ “black box” шинжтэй хэвээр байна. Иймээс бодит хэрэглээ болон судалгааны зорилгоос хамааран загварын сонголтыг тэнцвэртэй хийх шаардлагатай гэж дүгнэж байна.
Эцэст нь, энэхүү харьцуулалтаас харахад sentiment analysis судалгаанд нэг универсал шилдэг арга гэж байхгүй бөгөөд өгөгдлийн хэмжээ, домэйн, бодит хэрэглээний шаардлагаас хамааран энгийн baseline-ээс эхлээд гүн сургалтын загвар хүртэлх өргөн хүрээний аргуудыг уялдуулан ашиглах нь хамгийн үр дүнтэй хандлага болох нь тодорхой байна.





## 3. EMBEDDING АРГУУД БОЛОН ОНОЛЫН ҮНДЭС
Embedding аргууд нь текст өгөгдлийг машин сургалт болон гүн сургалтын загваруудад ашиглах боломжтой тоон вектор хэлбэрт дүрслэх зорилготой. Уламжлалт текстийн дүрслэл нь үгсийг бие даасан нэгж гэж үздэг бол embedding аргууд нь үг, өгүүлбэр, баримт бичгийн утга зүйн холбоог хадгалсан нягт (dense) вектор үүсгэхийг зорьдог. Эдгээр аргууд нь distributional semantics буюу “ижил орчинд хэрэглэгддэг үгс ижил утгатай” гэсэн онолын үндэст тулгуурладаг.

### 3.1 TF-IDF
TF-IDF (Term Frequency – Inverse Document Frequency) нь текстийг статистик шинжид суурилан дүрслэх уламжлалт арга юм. Уг арга нь тухайн үг нэг баримтад хэр давтамжтай гарч байгааг (TF), мөн нийт корпус дотор хэр түгээмэл байгааг (IDF) харгалзан үзэж жин оноодог. Үүний үр дүнд нэг баримтад чухал боловч бүх баримтад түгээмэл биш үгс илүү өндөр жинтэй болдог.
TF-IDF нь learned embedding биш боловч хурдан, тайлбарлахад хялбар тул sentiment analysis-д суурь (baseline) representation болгон өргөн ашиглагддаг.

### 3.2 Word2Vec – CBOW
Word2Vec-ийн Continuous Bag-of-Words (CBOW) загвар нь контекст үгсийг ашиглан төв үгийг таамаглах зарчмаар embedding сургадаг. Энэхүү загвар нь орчны үгсийг нэгтгэн ашигладаг тул тооцооллын хувьд хөнгөн, сургалтын хурд өндөр байдаг.
CBOW нь үгсийн дарааллыг харгалзан үздэггүй ч ижил орчинд хэрэглэгддэг үгсийг вектор орон зайд ойр байрлуулах чадвартай тул үгийн утга зүйн ерөнхий холбоог сайн илэрхийлдэг.

### 3.3 Word2Vec – Skip-gram
Skip-gram загвар нь CBOW-оос эсрэг чиглэлтэй бөгөөд төв үгийг ашиглан түүний орчны үгсийг таамагладаг. Энэ нь ховор үгсийн embedding-ийг илүү сайн суралцах давуу талтай боловч сургалтын хугацаа урт, тооцооллын зардал өндөр байдаг.
Skip-gram нь үгсийн нарийн утга зүйн ялгааг илүү сайн хадгалдаг тул их хэмжээний корпус дээр embedding сургахад тохиромжтой.

### 3.4 BERT (Base BERT)
BERT (Bidirectional Encoder Representations from Transformers) нь Transformer архитектурт суурилсан, контекстэд мэдрэмтгий embedding загвар юм. Base BERT нь 12 encoder давхарга, 768 хэмжээтэй hidden vector ашигладаг бөгөөд үгийн утгыг өмнөх болон дараах контекстийг зэрэг харгалзан ойлгодог.
BERT нь Masked Language Modeling болон Next Sentence Prediction даалгавруудаар урьдчилан сургагддаг бөгөөд sentiment analysis зэрэг олон NLP даалгаварт өндөр гүйцэтгэл үзүүлдэг.

### 3.5 RoBERTa
RoBERTa нь BERT-ийн архитектурыг хадгалсан боловч сургалтын стратегийг сайжруулсан загвар юм. Next Sentence Prediction-ийг хасч, илүү их өгөгдөл, урт хугацааны сургалт, динамик masking ашигласнаараа илүү бат бөх embedding сургадаг.
Практикт RoBERTa нь Base BERT-ээс илүү тогтвортой, өндөр гүйцэтгэл үзүүлэх нь олон судалгаагаар батлагдсан.

### 3.6 ALBERT
ALBERT нь BERT-ийн хөнгөн хувилбар бөгөөд параметрийн тоог багасгах зорилготой. Үүнд embedding factorization болон давхарга хоорондын parameter sharing ашигладаг.
ALBERT нь тооцооллын зардал бага, санах ой хэмнэлттэй боловч гүйцэтгэлийн хувьд зарим тохиолдолд BERT, RoBERTa-аас доогуур байж болдог.

### 3.7 HateBERT
HateBERT нь BERT загварыг үзэн ядалт, доромжлол ихтэй текст дээр дахин сургаж (domain-adaptive pretraining) гаргасан загвар юм. Энэ нь toxic language, hate speech зэрэг сөрөг илэрхийллийг илүү сайн ялгах чадвартай.
HateBERT нь домэйнд тохирсон embedding ашиглах нь зарим даалгаварт ерөнхий загвараас илүү үр дүнтэй байж болохыг харуулдаг.

### 3.8 SBERT (Sentence-BERT)
SBERT нь өгүүлбэрийн түвшний embedding үүсгэх зорилготойгоор BERT-д суурилан бүтээгдсэн загвар юм. Siamese архитектур ашигласнаар өгүүлбэр бүрийг тогтмол урттай вектор болгон хувиргаж, cosine similarity зэрэг хэмжүүрээр хурдан харьцуулах боломж олгодог.
SBERT нь sentence-level sentiment analysis, ижил төстэй өгүүлбэр хайлт зэрэг даалгаварт онцгой тохиромжтой.
## 4. ТУРШИЛТЫН ОРЧИН
	Энэхүү судалгаанд хийгдсэн туршилтуудыг машин сургалт болон гүн сургалтын ажлуудад тохирсон техник, програм хангамжийн орчинд гүйцэтгэсэн. Туршилтын ажлыг зөөврийн компьютер ашиглан хийсэн бөгөөд хэрэглэсэн техник хангамжийн үндсэн үзүүлэлтүүдийг доор дурдав.
• Процессор (CPU): Intel Core i7 
• Санах ой (RAM): 16 GB 
• График карт (GPU): NVIDIA GeForce RTX 4060 
• Үйлдлийн систем: Linux (Ubuntu суурьтай орчин)
 • Python хувилбар: Python 3.13 
Програм хангамжийн хувьд машин сургалт, гүн сургалт болон өгөгдөл боловсруулахад өргөн хэрэглэгддэг дараах сангуудыг ашигласан.
•	PyTorch – гүн сургалтын загварууд болон LSTM-ийг хэрэгжүүлэхэд
•	Transformers (HuggingFace) – Transformer суурьтай embedding загварууд (BERT, SBERT, RoBERTa, ALBERT, HateBERT) үүсгэхэд
•	Scikit-learn – Logistic Regression, Random Forest, AdaBoost зэрэг классик машин сургалтын загваруудыг хэрэгжүүлэхэд
•	Gensim – Word2Vec embedding (CBOW, Skip-gram) сургах болон ашиглахад
•	SciPy, NumPy, Pandas – өгөгдөл боловсруулах, тооцоолол хийхэд
Transformer суурьтай embedding загварууд (BERT, SBERT, ALBERT, HateBERT, RoBERTa)-ыг үүсгэх болон ашиглах үед GPU-г идэвхтэй ашигласан бол, TF-IDF болон Word2Vec-д суурилсан embedding болон классик машин сургалтын загваруудыг CPU орчинд гүйцэтгэсэн. Ийнхүү тооцооллын нөөцийг оновчтой хуваарилснаар сургалтын хугацааг багасгаж, туршилтыг тогтвортой гүйцэтгэх боломжийг хангаж өгсөн.
##5. Фолдерийн бүтэц (Project Folder Structure)
 Туршилтын бүх үр дүн, embedding болон загварын файлуудыг нэгдсэн, системтэй бүтэцтэйгээр хадгалсан. Үндсэн бүтэц нь дараах байдалтай.
 
                    

Зураг 1: Фолдерын ерөнхий бүтэц                



Зураг 2: Үр дүнгийн хадгалагдсан байдал
 
Энэхүү судалгаанд ашиглагдсан бүх туршилтын код, embedding, сургасан загвар болон гарсан үр дүнг нэгдсэн, системтэй фолдерийн бүтэцтэйгээр зохион байгуулан хадгалсан. Ингэснээр туршилтыг дахин давтах, өөр embedding эсвэл загвар нэмэх, мөн үр дүнг харьцуулахад хялбар болсон.
Зураг 1-д төслийн үндсэн фолдерийн ерөнхий бүтэц харагдана. Үндсэн NLP фолдер нь төслийн гол орчин бөгөөд дараах дэд фолдеруудаас бүрдэнэ. artifacts_1 фолдерт сургалтын явцад үүссэн завсрын файлууд болон хадгалсан загваруудыг байршуулсан. embed_and_run фолдер нь embedding үүсгэх, туршилтыг ажиллуулах үндсэн скриптүүдийг агуулна. Харин venv фолдер нь төслийн Python виртуал орчныг хадгалах зориулалттай.
Зураг 2-т embedding тус бүрийн туршилтын үр дүнг хадгалсан фолдеруудыг харуулсан. Энд albert_cv_top10_40runs, bert_base_cv_top10_40runs, roberta_cv_top10_40runs, sbert_cv_top10_40runs, hatebert_cv_top10_40runs зэрэг фолдерууд нь тухайн embedding аргыг ашиглан хийсэн cross-validation туршилтуудын үр дүнг агуулна. Мөн tfidf_cv_top10_40runs, w2v_cbow_cv_top10_40runs, w2v_sg_cv_top10_40runs фолдерууд нь TF-IDF болон Word2Vec (CBOW, Skip-gram) аргуудын туршилтын үр дүнг хадгалж байна.
Фолдерийн нэршилд ашиглагдсан cv нь cross-validation, top10 нь шилдэг 10 тохиргоог, 40runs нь туршилтыг 40 удаа давтан гүйцэтгэсэн болохыг илэрхийлнэ. Ийнхүү туршилтын үр дүнг embedding тус бүрээр ялган хадгалснаар өөр өөр аргуудын гүйцэтгэлийг шударгаар харьцуулах, мөн статистикийн хувьд тогтвортой дүгнэлт гаргах боломж бүрдсэн.
Энэхүү фолдерийн бүтэц нь судалгааны ажлын давтагдах чадвар (reproducibility) болон туршилтын менежментийг хангаж, төслийн код болон үр дүнг эмх цэгцтэй удирдах давуу талыг олгож байна.

## 6. ҮР ДҮН
## 6.1 TF-IDF + ангилагч загваруудын гүйцэтгэл
| C параметр | Logistic Regression (Acc / F1) | Random Forest (Acc / F1) | AdaBoost (Acc / F1) | LSTM (Acc / F1) |
| ---------: | -----------------------------: | -----------------------: | ------------------: | --------------: |
|       10.0 |                0.8291 / 0.8301 |          0.7824 / 0.7871 |     0.7695 / 0.7704 | 0.8129 / 0.8132 |
|        5.0 |                0.8287 / 0.8297 |          0.7763 / 0.7801 |     0.7695 / 0.7704 | 0.8122 / 0.8108 |
|        2.0 |                0.8273 / 0.8283 |          0.7813 / 0.7862 |     0.7695 / 0.7704 | 0.8057 / 0.8173 |
|        1.0 |                0.8248 / 0.8258 |          0.7852 / 0.7899 |     0.7695 / 0.7704 | 0.8094 / 0.8173 |
|        0.5 |                0.8216 / 0.8226 |          0.7815 / 0.7867 |     0.7695 / 0.7704 | 0.8132 / 0.8160 |
|        0.1 |                0.8144 / 0.8158 |          0.7793 / 0.7846 |     0.7695 / 0.7704 | 0.8109 / 0.8066 |
|       0.05 |                0.8122 / 0.8135 |          0.7827 / 0.7864 |     0.7695 / 0.7704 | 0.8124 / 0.8133 |
|       0.01 |                0.7983 / 0.7996 |          0.7795 / 0.7835 |     0.7695 / 0.7704 | 0.8136 / 0.8186 |


Энэхүү хүснэгтэд C параметрийн өөрчлөлт бүрийн үед ангилагч загваруудын гүйцэтгэлийг Accuracy болон F1-score-оор харьцуулан үзүүлсэн. Logistic Regression загварын хувьд C өндөр үед гүйцэтгэл сайжирч, C багасах тусам регуляризаци ихэсч гүйцэтгэл буурч байна. Random Forest болон AdaBoost загваруудын гүйцэтгэл C параметрээс бараг хамааралгүй, тогтвортой байна. Харин LSTM загвар нь C-тэй шууд хамааралгүй боловч бүх тохиргоонд ойролцоо, харьцангуй сайн гүйцэтгэл үзүүлж байна.
### 6.2 Word2Vec (CBOW) + ангилагч загваруудын гүйцэтгэл
| C параметр | Logistic Regression (Acc / F1) | Random Forest (Acc / F1) | AdaBoost (Acc / F1) | LSTM (Acc / F1) |
| ---------: | -----------------------------: | -----------------------: | ------------------: | --------------: |
|       10.0 |                0.8402 / 0.8414 |          0.8069 / 0.8120 |     0.8026 / 0.8054 | 0.8350 / 0.8404 |
|        5.0 |                0.8404 / 0.8416 |          0.8104 / 0.8152 |     0.8026 / 0.8054 | 0.8357 / 0.8377 |
|        2.0 |                0.8407 / 0.8418 |          0.8072 / 0.8125 |     0.8026 / 0.8054 | 0.8351 / 0.8370 |
|        1.0 |                0.8401 / 0.8412 |          0.8088 / 0.8140 |     0.8026 / 0.8054 | 0.8344 / 0.8400 |
|        0.5 |                0.8399 / 0.8411 |          0.8118 / 0.8160 |     0.8026 / 0.8054 | 0.8361 / 0.8360 |
|        0.1 |                0.8383 / 0.8398 |          0.8058 / 0.8109 |     0.8026 / 0.8054 | 0.8349 / 0.8378 |
|       0.05 |                0.8375 / 0.8392 |          0.8079 / 0.8129 |     0.8026 / 0.8054 | 0.8328 / 0.8297 |
|       0.01 |                0.8312 / 0.8334 |          0.8094 / 0.8143 |     0.8026 / 0.8054 | 0.8331 / 0.8341 |


Энэхүү хүснэгтэд C параметрийн өөрчлөлт бүрийн үед ангилагч загваруудын Accuracy болон F1-score-ийг нэгтгэн харуулсан. Logistic Regression болон LSTM загварууд бүх C утгад тогтвортой, ойролцоо гүйцэтгэл үзүүлсэн бол Random Forest дунд зэргийн үр дүнтэй байна. AdaBoost загварын гүйцэтгэл C параметрээс хамааралгүй, бүх тохиргоонд ижил байгаа нь ажиглагдаж байна.
### 6.3 Word2Vec (Skip-gram) + ангилагч загваруудын гүйцэтгэл
| C параметр | Logistic Regression (Acc / F1) | Random Forest (Acc / F1) | AdaBoost (Acc / F1) | LSTM (Acc / F1) |
| ---------: | -----------------------------: | -----------------------: | ------------------: | --------------: |
|       10.0 |                0.8555 / 0.8569 |          0.8272 / 0.8314 |     0.8142 / 0.8166 | 0.8471 / 0.8517 |
|        5.0 |                0.8561 / 0.8574 |          0.8252 / 0.8299 |     0.8142 / 0.8166 | 0.8479 / 0.8490 |
|        2.0 |                0.8556 / 0.8569 |          0.8237 / 0.8283 |     0.8142 / 0.8166 | 0.8488 / 0.8473 |
|        1.0 |                0.8552 / 0.8567 |          0.8246 / 0.8290 |     0.8142 / 0.8166 | 0.8468 / 0.8474 |
|        0.5 |                0.8541 / 0.8555 |          0.8215 / 0.8257 |     0.8142 / 0.8166 | 0.8466 / 0.8429 |
|        0.1 |                0.8480 / 0.8497 |          0.8253 / 0.8289 |     0.8142 / 0.8166 | 0.8496 / 0.8522 |
|       0.05 |                0.8421 / 0.8439 |          0.8280 / 0.8323 |     0.8142 / 0.8166 | 0.8445 / 0.8498 |
|       0.01 |                0.8146 / 0.8167 |          0.8253 / 0.8294 |     0.8142 / 0.8166 | 0.8455 / 0.8506 |


Энэхүү хүснэгтэд C параметрийн өөрчлөлт бүрийн үед ангилагч загваруудын Accuracy болон F1-score-ийг нэгтгэн харуулсан. Logistic Regression загвар C = 1.0–10.0 үед хамгийн өндөр, тогтвортой гүйцэтгэл үзүүлсэн бол Random Forest дунд зэргийн үр дүнтэй байна. AdaBoost загвар бүх C утгад ижил гүйцэтгэлтэй байгаа нь уг параметртэй хамааралгүйг харуулж байна. LSTM загвар нь C параметрээс үл хамааран тогтвортой, харьцангуй өндөр F1-score үзүүлжээ.
### 6.4 Хүснэгт: BERT (base) + ангилагч загваруудын гүйцэтгэл
| C параметр | Logistic Regression (Acc / F1) | Random Forest (Acc / F1) | AdaBoost (Acc / F1) | LSTM (Acc / F1) |
| ---------: | -----------------------------: | -----------------------: | ------------------: | --------------: |
|        1.0 |                0.7849 / 0.7860 |          0.7163 / 0.7138 |     0.7250 / 0.7239 | 0.7645 / 0.7663 |
|        0.5 |                0.7849 / 0.7858 |          0.7177 / 0.7149 |     0.7250 / 0.7239 | 0.7563 / 0.7367 |
|        2.0 |                0.7847 / 0.7859 |          0.7176 / 0.7146 |     0.7250 / 0.7239 | 0.7616 / 0.7547 |
|        5.0 |                0.7835 / 0.7848 |          0.7204 / 0.7178 |     0.7250 / 0.7239 | 0.7542 / 0.7358 |
|       10.0 |                0.7832 / 0.7845 |          0.7188 / 0.7154 |     0.7250 / 0.7239 | 0.7642 / 0.7648 |
|        0.1 |                0.7845 / 0.7848 |          0.7074 / 0.7037 |     0.7250 / 0.7239 | 0.7563 / 0.7431 |
|       0.05 |                0.7824 / 0.7826 |          0.7173 / 0.7142 |     0.7250 / 0.7239 | 0.7618 / 0.7714 |
|       0.01 |                0.7703 / 0.7696 |          0.7191 / 0.7160 |     0.7250 / 0.7239 | 0.7551 / 0.7756 |

BERT (base) embedding ашигласан тохиолдолд ангилагч загваруудын гүйцэтгэл C параметрээс ихээхэн хамааралгүй, ерөнхийдөө тогтвортой байна. Logistic Regression загвар бүх C утгад ойролцоо үр дүн (Accuracy ≈ 0.78, F1 ≈ 0.78) үзүүлж, C-ийг хэт багасгахад (0.01) л бага зэрэг уналт ажиглагдаж байна.
Random Forest болон AdaBoost загваруудын гүйцэтгэл харьцангуй доогуур бөгөөд C параметр өөрчлөгдөхөд мэдэгдэхүйц сайжрал ажиглагдахгүй байна. Ялангуяа AdaBoost бүх тохиргоонд бараг ижил үр дүн үзүүлсэн нь C параметр уг загварт нөлөөлөхгүйг илтгэнэ.
LSTM загварын хувьд BERT embedding-тэй хослуулсан үед Logistic Regression-тэй ойролцоо гүйцэтгэл үзүүлж байгаа ч тогтворжилт харьцангуй сул, C утгаас хамааран F1-score хэлбэлзэж байна.
Ерөнхийд нь дүгнэвэл, BERT (base) embedding нь загваруудын ялгааг багасгаж, C параметрийн нөлөөг сулруулж байгаа бөгөөд энэ нөхцөлд ангилагчийн төрөлөөс илүү embedding-ийн чанар гүйцэтгэлд голлон нөлөөлж байна.

### 6.5 Хүснэгт: SBERT + ангилагч загваруудын гүйцэтгэл
| C параметр | Logistic Regression (Acc / F1) | Random Forest (Acc / F1) | AdaBoost (Acc / F1) | LSTM (Acc / F1) |
| ---------: | -----------------------------: | -----------------------: | ------------------: | --------------: |
|       10.0 |                0.8291 / 0.8301 |          0.7824 / 0.7871 |     0.7695 / 0.7704 | 0.8129 / 0.8132 |
|        5.0 |                0.8287 / 0.8297 |          0.7763 / 0.7801 |     0.7695 / 0.7704 | 0.8122 / 0.8108 |
|        2.0 |                0.8273 / 0.8283 |          0.7813 / 0.7862 |     0.7695 / 0.7704 | 0.8057 / 0.8173 |
|        1.0 |                0.8248 / 0.8258 |          0.7852 / 0.7899 |     0.7695 / 0.7704 | 0.8094 / 0.8173 |
|        0.5 |                0.8216 / 0.8226 |          0.7815 / 0.7867 |     0.7695 / 0.7704 | 0.8132 / 0.8160 |
|        0.1 |                0.8144 / 0.8158 |          0.7793 / 0.7846 |     0.7695 / 0.7704 | 0.8109 / 0.8066 |
|       0.05 |                0.8122 / 0.8135 |          0.7827 / 0.7864 |     0.7695 / 0.7704 | 0.8124 / 0.8133 |
|       0.01 |                0.7983 / 0.7996 |          0.7795 / 0.7835 |     0.7695 / 0.7704 | 0.8136 / 0.8186 |

SBERT embedding ашигласан үед загваруудын гүйцэтгэл ерөнхийдөө тогтвортой байна. Logistic Regression нь C = 5–10 орчимд хамгийн сайн үр дүн (Accuracy ≈ 0.83, F1 ≈ 0.83) үзүүлж, C багасах тусам гүйцэтгэл аажмаар буурч байна.
Random Forest болон AdaBoost загваруудын гүйцэтгэл C-ээс үл хамааран бараг өөрчлөгдөхгүй байгаа нь эдгээр загварууд SBERT embedding-тэй үед параметрт мэдрэмтгий бус байгааг харуулж байна.
LSTM загвар нь Logistic Regression-тэй ойролцоо, зарим тохиргоонд арай өндөр F1-score үзүүлсэн ч C параметртэй шууд хамаарал ажиглагдахгүй байна.
Ерөнхийд нь дүгнэвэл, SBERT embedding нь загваруудын гүйцэтгэлийг тогтворжуулж, C параметрийн нөлөөг багасгаж байгаа бөгөөд энэ нөхцөлд ангилагчийн төрөл бус, embedding-ийн чанар илүү чухал нөлөө үзүүлж байна.
### 6.6 Хүснэгт: RoBERTa + ангилагч загваруудын гүйцэтгэл
| C параметр | Logistic Regression (Acc / F1) | Random Forest (Acc / F1) | AdaBoost (Acc / F1) | LSTM (Acc / F1) |
| ---------: | -----------------------------: | -----------------------: | ------------------: | --------------: |
|       10.0 |                0.8423 / 0.8431 |          0.8071 / 0.8089 |     0.8012 / 0.7981 | 0.8242 / 0.8321 |
|        5.0 |                0.8411 / 0.8418 |          0.7984 / 0.8001 |     0.8012 / 0.7981 | 0.8277 / 0.8293 |
|        2.0 |                0.8385 / 0.8391 |          0.8067 / 0.8091 |     0.8012 / 0.7981 | 0.8264 / 0.8315 |
|        1.0 |                0.8374 / 0.8380 |          0.8054 / 0.8071 |     0.8012 / 0.7981 | 0.8236 / 0.8326 |
|        0.5 |                0.8342 / 0.8348 |          0.8076 / 0.8092 |     0.8012 / 0.7981 | 0.8253 / 0.8322 |
|        0.1 |                0.8290 / 0.8297 |          0.8029 / 0.8044 |     0.8012 / 0.7981 | 0.8253 / 0.8255 |
|       0.05 |                0.8228 / 0.8233 |          0.7997 / 0.8011 |     0.8012 / 0.7981 | 0.8247 / 0.8241 |
|       0.01 |                0.8079 / 0.8083 |          0.7997 / 0.8011 |     0.8012 / 0.7981 | 0.8212 / 0.8119 |

RoBERTa embedding ашигласан үед бүх ангилагч загваруудын гүйцэтгэл харьцангуй өндөр бөгөөд тогтвортой байна. Logistic Regression загвар нь C = 5–10 орчимд хамгийн сайн үр дүн (Accuracy ≈ 0.84, F1 ≈ 0.84) үзүүлж, C багасах тусам гүйцэтгэл аажмаар буурч байна.
Random Forest болон AdaBoost загваруудын гүйцэтгэл C параметрээс бараг хамааралгүй, ойролцоогоор тогтвортой түвшинд хадгалагдсан байна.
LSTM загварын хувьд Logistic Regression-тэй ойролцоо, зарим тохиргоонд илүү өндөр F1-score үзүүлж байгаа ч C параметртэй шууд хамаарал ажиглагдахгүй байна.
Ерөнхийд нь дүгнэвэл, RoBERTa embedding нь загваруудын ялгааг багасгаж, C параметрийн нөлөөг сулруулж байгаа бөгөөд энэ нөхцөлд гүйцэтгэлд embedding-ийн чанар голлох үүрэг гүйцэтгэж байна.
### 6.7 Хүснэгт: HateBERT + ангилагч загваруудын гүйцэтгэл
| C параметр | Logistic Regression (Acc / F1) | Random Forest (Acc / F1) | AdaBoost (Acc / F1) | LSTM (Acc / F1) |
| ---------: | -----------------------------: | -----------------------: | ------------------: | --------------: |
|       0.05 |                0.8178 / 0.8184 |          0.7605 / 0.7630 |     0.7616 / 0.7601 | 0.8022 / 0.8122 |
|        0.1 |                0.8178 / 0.8185 |          0.7643 / 0.7669 |     0.7616 / 0.7601 | 0.7982 / 0.7858 |
|        0.5 |                0.8191 / 0.8195 |          0.7665 / 0.7689 |     0.7616 / 0.7601 | 0.8014 / 0.8148 |
|        5.0 |                0.8183 / 0.8190 |          0.7640 / 0.7673 |     0.7616 / 0.7601 | 0.8076 / 0.8077 |
|        2.0 |                0.8187 / 0.8192 |          0.7592 / 0.7617 |     0.7616 / 0.7601 | 0.8058 / 0.8087 |
|       10.0 |                0.8184 / 0.8191 |          0.7634 / 0.7663 |     0.7616 / 0.7601 | 0.8097 / 0.8107 |
|        1.0 |                0.8198 / 0.8203 |          0.7602 / 0.7640 |     0.7616 / 0.7601 | 0.8047 / 0.8128 |
|       0.01 |                0.8111 / 0.8116 |          0.7612 / 0.7643 |     0.7616 / 0.7601 | 0.8029 / 0.7953 |


HateBERT embedding ашигласан үед бүх загваруудын гүйцэтгэл ойролцоо, тогтвортой байна. Logistic Regression загвар C = 0.5–1.0 орчимд хамгийн сайн үр дүн (Accuracy ≈ 0.82, F1 ≈ 0.82) үзүүлж, C хэт багасахад (0.01) гүйцэтгэл бага зэрэг буурч байна.
Random Forest болон AdaBoost загваруудын гүйцэтгэл C параметрээс үл хамааран бараг өөрчлөгдөхгүй байгаа нь эдгээр загварууд C-д мэдрэг бус байгааг харуулж байна.
LSTM загварын хувьд Logistic Regression-тэй ойролцоо эсвэл арай өндөр F1-score үзүүлж байгаа ч C параметртэй шууд хамаарал ажиглагдахгүй байна.
Ерөнхийд нь дүгнэвэл, HateBERT нь C параметрийн нөлөөг багасгаж, загваруудын гүйцэтгэлийг тогтвортой болгож байгаа бөгөөд энэ нөхцөлд ангилагчийн төрөл бус, embedding-ийн домэйнд тохирсон байдал илүү чухал нөлөө үзүүлж байна.
### 6.8 ALBERT + ангилагч загваруудын гүйцэтгэл
| C параметр | Logistic Regression (Acc / F1) | Random Forest (Acc / F1) | AdaBoost (Acc / F1) | LSTM (Acc / F1) |
| ---------: | -----------------------------: | -----------------------: | ------------------: | --------------: |
|        0.5 |                0.8080 / 0.8075 |          0.7409 / 0.7402 |     0.7376 / 0.7315 | 0.7845 / 0.7913 |
|        1.0 |                0.8075 / 0.8069 |          0.7382 / 0.7388 |     0.7376 / 0.7315 | 0.7869 / 0.7839 |
|        5.0 |                0.8072 / 0.8067 |          0.7367 / 0.7360 |     0.7376 / 0.7315 | 0.7804 / 0.7658 |
|       10.0 |                0.8069 / 0.8065 |          0.7410 / 0.7399 |     0.7376 / 0.7315 | 0.7612 / 0.7236 |
|        2.0 |                0.8070 / 0.8065 |          0.7412 / 0.7403 |     0.7376 / 0.7315 | 0.7652 / 0.7908 |
|        0.1 |                0.8099 / 0.8092 |          0.7408 / 0.7404 |     0.7376 / 0.7315 | 0.7842 / 0.7944 |
|       0.05 |                0.8104 / 0.8094 |          0.7356 / 0.7362 |     0.7376 / 0.7315 | 0.7737 / 0.7563 |
|       0.01 |                0.8094 / 0.8087 |          0.7405 / 0.7390 |     0.7376 / 0.7315 | 0.7811 / 0.7927 |


Энэхүү хүснэгтээс харахад ALBERT embedding ашигласан үед загваруудын гүйцэтгэл харьцангуй доогуур боловч тогтвортой байна. Logistic Regression загвар бүх C утгад ойролцоо үр дүн (Accuracy ≈ 0.81, F1 ≈ 0.81) үзүүлж, C параметрээс бараг хамаарахгүй байгааг харуулж байна.
Random Forest болон AdaBoost загваруудын гүйцэтгэл мөн тогтвортой боловч Logistic Regression-ээс доогуур байна.
LSTM загварын хувьд бусад ангилагчтай харьцуулахад илүү өндөр гүйцэтгэл үзүүлж байгаа ч C ихсэхэд (ялангуяа 10.0) гүйцэтгэл буурч байгаа нь overfitting эсвэл сургалтын тогтворгүй байдал байж болохыг илтгэнэ.
Ерөнхийд нь дүгнэвэл, ALBERT нь хөнгөн загварын хувьд тогтвортой боловч BERT, RoBERTa зэрэг загваруудаас гүйцэтгэл доогуур, мөн энэ нөхцөлд C параметрийн нөлөө харьцангуй бага байгааг уг үр дүн харуулж байна.


## 7. Ерөнхий дүгнэлт
Энэхүү судалгааны хүрээнд TF-IDF болон орчин үеийн embedding аргууд (BERT, SBERT, RoBERTa, HateBERT, ALBERT)-ыг төрөл бүрийн ангилагч загваруудтай хослуулан туршиж үзэхэд өгөгдлийн дүрслэл (embedding / representation) нь загварын гүйцэтгэлд хамгийн их нөлөөтэй хүчин зүйл болох нь тодорхой ажиглагдлаа.
TF-IDF дээр суурилсан туршилтуудад Logistic Regression загвар зөв C параметртэй үед хамгийн өндөр, тогтвортой гүйцэтгэл үзүүлсэн нь өндөр хэмжээст, sparse өгөгдөлд шугаман загварууд тохиромжтойг баталж байна. Энэ тохиолдолд C параметрийн нөлөө тод илэрч, хэт их утга нь overfitting, хэт бага утга нь underfitting үүсгэж байв.
Харин BERT, SBERT, RoBERTa, HateBERT, ALBERT зэрэг contextual embedding ашигласан үед C параметрийн нөлөө мэдэгдэхүйц багасч, загваруудын гүйцэтгэл илүү тогтвортой болсон. Энэ нь embedding өөрөө текстийн утга зүйн мэдээллийн ихэнхийг агуулж чадсан тул ангилагч загварын hyperparameter-ууд хоёрдогч үүрэг гүйцэтгэж эхэлснийг харуулж байна.
Embedding аргуудын хооронд харьцуулбал:
•	RoBERTa болон SBERT нь хамгийн өндөр, тогтвортой гүйцэтгэл үзүүлсэн
•	BERT (base) нь найдвартай боловч RoBERTa-аас бага зэрэг доогуур
•	HateBERT нь домэйнд тохирсон өгөгдөлд тогтвортой, дундаж гүйцэтгэлтэй
•	ALBERT нь тооцооллын хувьд хөнгөн ч гүйцэтгэл харьцангуй доогуур байсан
Ангилагч загваруудын хувьд:
•	Logistic Regression бараг бүх embedding дээр хамгийн тогтвортой, найдвартай үр дүн өгсөн
•	Random Forest болон AdaBoost нь текстийн embedding өгөгдөлд харьцангуй сул гүйцэтгэлтэй
•	LSTM нь embedding-д суурилсан үед зарим тохиолдолд Logistic Regression-тэй ойролцоо эсвэл илүү F1-score үзүүлсэн ч сургалтын тогтвортой байдал харьцангуй сул байв
Дүгнэж хэлбэл, sentiment analysis-ийн гүйцэтгэлд ямар загвар ашигласнаас илүүтэйгээр ямар embedding ашигласан нь шийдвэрлэх нөлөөтэй бөгөөд сайн embedding ашигласан нөхцөлд энгийн ангилагч ч өндөр үр дүн үзүүлэх боломжтой болохыг энэхүү судалгаа тодорхой харуулж байна. Иймээс бодит хэрэглээнд тооцооллын нөөц, тайлбарлах шаардлага, өгөгдлийн шинж чанарыг харгалзан embedding ба ангилагчийг тэнцвэртэй сонгох нь хамгийн оновчтой хандлага гэж дүгнэж байна.
Судалгааны хязгаарлалт ба цаашдын судалгааны чиглэл
Энэхүү судалгааны ажлын үр дүнг тайлбарлахдаа тодорхой хязгаарлалтуудыг харгалзан үзэх шаардлагатай. Нэгдүгээрт, судалгаанд ашигласан өгөгдөл нь голчлон IMDB кино шүүмжид төвлөрсөн тул үр дүн нь бусад домэйнд (жишээ нь сошиал сүлжээ, мэдээ, бүтээгдэхүүний үнэлгээ) шууд ерөнхийлөгдөх боломж хязгаарлагдмал байна. Үүнтэй холбоотойгоор зарим embedding загварууд, тухайлбал HateBERT, нь тухайн домэйнд илүү тохиромжтой боловч IMDB өгөгдөл дээр давуу тал нь бүрэн илэрч чадаагүй байж болох юм.
Хоёрдугаарт, судалгаанд ашигласан ихэнх туршилтууд баримт бичгийн түвшний sentiment analysis-д чиглэсэн тул нэг шүүмж доторх олон талт сэтгэл хөдлөлийг нарийвчлан илрүүлэх боломж хязгаарлагдмал байна. Мөн hyperparameter тохируулгыг C параметр болон зарим суурь тохиргоогоор хязгаарласан нь илүү өргөн хайлтын орон зайг хамарч чадаагүй байж болзошгүй.
Гуравдугаарт, гүн сургалтын загваруудын хувьд гүйцэтгэлийг голчлон accuracy болон F1-score-оор үнэлсэн бөгөөд загварын шийдвэр гаргалтын тайлбарлах чадвар тусгайлан судлагдаагүй нь энэхүү ажлын нэг сул тал болж байна. Ялангуяа contextual embedding ашигласан загварууд “black-box” шинжтэй тул бодит хэрэглээнд нэвтрүүлэхэд нэмэлт судалгаа шаардлагатай.
Дээрх хязгаарлалтуудад үндэслэн цаашдын судалгаанд хэд хэдэн чиглэлийг санал болгож байна. Юуны өмнө, RoBERTa, SBERT зэрэг өндөр гүйцэтгэлтэй embedding загваруудыг тухайн домэйнд дахин сургах (domain-specific fine-tuning) замаар гүйцэтгэлийг цаашид сайжруулах боломжтой. Мөн судалгааг sentence-level болон aspect-level sentiment analysis руу өргөтгөснөөр илүү нарийн, хэрэглээний ач холбогдолтой үр дүн гаргах боломж бүрдэнэ.
Цаашлаад, олон загварыг нэгтгэсэн ensemble арга болон real-time өгөгдөл дээрх туршилтыг хийснээр бодит системд ашиглах боломжийг нэмэгдүүлэх шаардлагатай гэж үзэж байна. Эцэст нь, SHAP, LIME зэрэг explainability аргуудыг ашиглан загварын шийдвэрийг тайлбарлах нь судалгааны чанар болон практик ач холбогдлыг цаашид нэмэгдүүлэх чухал чиглэл болох юм.
## 8. ЭХ СУРВАЛЖ
Learning Word Vectors for Sentiment Analysis
https://aclanthology.org/P11-1015/
•	Baselines and Bigrams: Simple, Good Sentiment and Topic Classification
https://aclanthology.org/P12-2018/
•	Efficient Estimation of Word Representations in Vector Space
https://arxiv.org/abs/1301.3781
•	Distributed Representations of Sentences and Documents
https://proceedings.mlr.press/v32/le14.html
•	Attention Is All You Need
https://arxiv.org/abs/1706.03762
•	BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
https://arxiv.org/abs/1810.04805
•	RoBERTa: A Robustly Optimized BERT Pretraining Approach
https://arxiv.org/abs/1907.11692
•	ALBERT: A Lite BERT for Self-Supervised Learning of Language Representations
https://arxiv.org/abs/1909.11942
•	Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
https://arxiv.org/abs/1908.10084
•	HateBERT: Retraining BERT for Abusive Language Detection in English
https://aclanthology.org/2020.alw-1.3/
•	Convolutional Neural Networks for Sentence Classification
https://arxiv.org/abs/1408.5882
•	Speech and Language Processing (3rd ed.)
https://web.stanford.
aclanthology.org
