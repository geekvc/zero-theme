<h2 id="nms主要目的">Nms主要目的</h2>
<p>在物体检测非极大值抑制应用十分广泛，主要目的是为了消除多余的框，找到最佳的物体检测的位置。</p>
<ul>
<li>等待</li>
<li></li>
</ul>
<p>如上图中：虽然几个框都检测到了人脸，但是我不需要这么多的框，我需要找到一个最能表达人脸的框。下图汽车检测也是同样的原理。</p>
<p>[图片上传失败…(image-a0c46b-1512223034395)]</p>
<p>##3领域情况非极大值抑制</p>
<p>3邻域是指判断该点是否比左边的一个点以及右边的一个点的数值大</p>
<p>[图片上传失败…(image-79154b-1512223034395)]</p>
<p>代码简单分析：对于一个从I[0]到I[W-1]的输入序列，对于有左邻和右邻的是从1到W-2，所以循环的初始位置是1。</p>
<p>（1）代码的3-5行是判断当前点是否左邻和右邻的值大，如果大的话该点就是极大值点。对于这样的点我们就已经知道i+1位置的值比i的小，所以对i+1的位置就不需要处理，所以可以直接处理i+2位置。对应于代码的第12行。</p>
<p>[图片上传失败…(image-b63d83-1512223034395)]</p>
<p>（2）如果在第3行代码中，不满足条件，那么该点的右邻就作为候选，对于代码第7行。循环的使用这一条件，候选就会采用单调递增的方式一直向右查找，直到找到满足大于右邻的点（对应于代码8-9行），若该点不是最右的点，则满足条件，为极大值点（对应于代码的10-11行）。</p>
<p>[图片上传失败…(image-337886-1512223034395)]</p>
<h2 id="原理">原理</h2>
<p>非极大值抑制，顾名思义就是把非极大值过滤掉（抑制）。下面我就R-CNN或者SPP_net中的matlab源码来进行解释。</p>
<pre class=" language-matlab"><code class="prism  language-matlab">
<span class="token keyword">function</span> picks <span class="token operator">=</span> <span class="token function">nms_multiclass</span><span class="token punctuation">(</span>boxes<span class="token punctuation">,</span> overlap<span class="token punctuation">)</span>

<span class="token comment" spellcheck="true">%%boxes为一个m*n的矩阵，其中m为boundingbox的个数，n的前4列为每个boundingbox的坐标，格式为</span>

<span class="token comment" spellcheck="true">%%（x1,y1,x2,y2）；第5:n列为每一类的置信度，一共n-5+1个置信度</span>

<span class="token comment" spellcheck="true">%%overlap为设定值，0.3,0.5 .....</span>

x1 <span class="token operator">=</span> <span class="token function">boxes</span><span class="token punctuation">(</span><span class="token operator">:</span><span class="token punctuation">,</span><span class="token number">1</span><span class="token punctuation">)</span><span class="token punctuation">;</span><span class="token comment" spellcheck="true">%所有boundingbox的x1坐标</span>

y1 <span class="token operator">=</span> <span class="token function">boxes</span><span class="token punctuation">(</span><span class="token operator">:</span><span class="token punctuation">,</span><span class="token number">2</span><span class="token punctuation">)</span><span class="token punctuation">;</span><span class="token comment" spellcheck="true">%所有boundingbox的y1坐标</span>

x2 <span class="token operator">=</span> <span class="token function">boxes</span><span class="token punctuation">(</span><span class="token operator">:</span><span class="token punctuation">,</span><span class="token number">3</span><span class="token punctuation">)</span><span class="token punctuation">;</span><span class="token comment" spellcheck="true">%所有boundingbox的x2坐标</span>

y2 <span class="token operator">=</span> <span class="token function">boxes</span><span class="token punctuation">(</span><span class="token operator">:</span><span class="token punctuation">,</span><span class="token number">4</span><span class="token punctuation">)</span><span class="token punctuation">;</span><span class="token comment" spellcheck="true">%所有boundingbox的y2坐标</span>

area <span class="token operator">=</span> <span class="token punctuation">(</span>x2<span class="token operator">-</span>x1<span class="token operator">+</span><span class="token number">1</span><span class="token punctuation">)</span> <span class="token operator">.*</span> <span class="token punctuation">(</span>y2<span class="token operator">-</span>y1<span class="token operator">+</span><span class="token number">1</span><span class="token punctuation">)</span><span class="token punctuation">;</span> <span class="token comment" spellcheck="true">%每个%所有boundingbox的面积</span>

<span class="token comment" spellcheck="true">%可能是从零开始，所以要加1</span>

picks <span class="token operator">=</span> <span class="token function">cell</span><span class="token punctuation">(</span><span class="token function">size</span><span class="token punctuation">(</span>boxes<span class="token punctuation">,</span> <span class="token number">2</span><span class="token punctuation">)</span><span class="token operator">-</span><span class="token number">4</span><span class="token punctuation">,</span> <span class="token number">1</span><span class="token punctuation">)</span><span class="token punctuation">;</span><span class="token comment" spellcheck="true">%为每一类预定义一个将要保留的cell</span>

<span class="token comment" spellcheck="true">%cell是一种很牛逼的数据类型，每个单元可存储任何数据，矩阵、传递函数、自定义类型</span>

<span class="token comment" spellcheck="true">%创建</span>

a<span class="token operator">=</span><span class="token function">cell</span><span class="token punctuation">(</span>n<span class="token punctuation">,</span>m<span class="token punctuation">)</span> 

那么就把a初始化为一个n行m列的空cell类型数据，

预分配内存

<span class="token comment" spellcheck="true">%读取内容:{下标}和(下标)  区别在于类型()是cell数组 ,{}是实际类型.结果显示是一致的，下标从1开始</span>

a<span class="token operator">=</span><span class="token function">cell</span><span class="token punctuation">(</span><span class="token number">2</span><span class="token punctuation">,</span><span class="token number">2</span><span class="token punctuation">)</span><span class="token punctuation">;</span><span class="token comment" spellcheck="true">%预分配</span>

a<span class="token punctuation">{</span><span class="token number">1</span><span class="token punctuation">,</span><span class="token number">1</span><span class="token punctuation">}</span><span class="token operator">=</span><span class="token string">'cellclass'</span><span class="token punctuation">;</span>

a<span class="token punctuation">{</span><span class="token number">1</span><span class="token punctuation">,</span><span class="token number">2</span><span class="token punctuation">}</span><span class="token operator">=</span><span class="token punctuation">[</span><span class="token number">1</span> <span class="token number">2</span> <span class="token number">2</span><span class="token punctuation">]</span><span class="token punctuation">;</span>

a<span class="token punctuation">{</span><span class="token number">2</span><span class="token punctuation">,</span><span class="token number">1</span><span class="token punctuation">}</span><span class="token operator">=</span><span class="token punctuation">[</span><span class="token string">'a'</span><span class="token punctuation">,</span><span class="token string">'b'</span><span class="token punctuation">,</span><span class="token string">'c'</span><span class="token punctuation">]</span><span class="token punctuation">;</span>

a<span class="token punctuation">{</span><span class="token number">2</span><span class="token punctuation">,</span><span class="token number">2</span><span class="token punctuation">}</span><span class="token operator">=</span><span class="token punctuation">[</span><span class="token number">9</span> <span class="token number">5</span> <span class="token number">6</span><span class="token punctuation">]</span><span class="token punctuation">;</span>

<span class="token comment" spellcheck="true">%以上创建一个</span>

%

<span class="token function">size</span><span class="token punctuation">(</span>X<span class="token punctuation">,</span><span class="token number">1</span><span class="token punctuation">)</span><span class="token punctuation">,</span>返回矩阵X的行数；

<span class="token comment" spellcheck="true">%size(X,2),返回矩阵X的列数；</span>

（<span class="token number">1</span>）s<span class="token operator">=</span><span class="token function">size</span><span class="token punctuation">(</span>A<span class="token punctuation">)</span><span class="token punctuation">,</span>

当只有一个输出参数时，返回一个行向量，该行向量的第一个元素时矩阵的行数，第二个元素是矩阵的列数。

（<span class="token number">2</span>）<span class="token punctuation">[</span>r<span class="token punctuation">,</span>c<span class="token punctuation">]</span><span class="token operator">=</span><span class="token function">size</span><span class="token punctuation">(</span>A<span class="token punctuation">)</span><span class="token punctuation">,</span>

当有两个输出参数时，size函数将矩阵的行数返回到第一个输出变量r，将矩阵的列数返回到第二个输出变量c。

（<span class="token number">3</span>）<span class="token function">size</span><span class="token punctuation">(</span>A<span class="token punctuation">,</span>n<span class="token punctuation">)</span>如果在size函数的输入参数中再添加一项n，并用<span class="token number">1</span>或<span class="token number">2</span>为n赋值，则 size将返回矩阵的行数或列数。其中r<span class="token operator">=</span><span class="token function">size</span><span class="token punctuation">(</span>A<span class="token punctuation">,</span><span class="token number">1</span><span class="token punctuation">)</span>该语句返回的时矩阵A的行数， c<span class="token operator">=</span><span class="token function">size</span><span class="token punctuation">(</span>A<span class="token punctuation">,</span><span class="token number">2</span><span class="token punctuation">)</span> 该语句返回的时矩阵A的列数。

<span class="token keyword">for</span> iS <span class="token operator">=</span> <span class="token number">5</span><span class="token operator">:</span><span class="token function">size</span><span class="token punctuation">(</span>boxes<span class="token punctuation">,</span> <span class="token number">2</span><span class="token punctuation">)</span><span class="token comment" spellcheck="true">%每一类单独进行</span>

    s <span class="token operator">=</span> <span class="token function">boxes</span><span class="token punctuation">(</span><span class="token operator">:</span><span class="token punctuation">,</span>iS<span class="token punctuation">)</span><span class="token punctuation">;</span>

    <span class="token punctuation">[</span><span class="token operator">~</span><span class="token punctuation">,</span> I<span class="token punctuation">]</span> <span class="token operator">=</span> <span class="token function">sort</span><span class="token punctuation">(</span>s<span class="token punctuation">)</span><span class="token punctuation">;</span><span class="token comment" spellcheck="true">%置信度从低到高排序，只要s的索引，赋值给I，~表示占位 表示函数输出的某个值在接下来并不使用</span>

<span class="token comment" spellcheck="true">%[B,ind]=sort(A)，计算后，B是A排序后的向量，A保持不变，ind是B中每一项对应于A 中项的索引。排序是安升序进行的</span>

    pick <span class="token operator">=</span> s<span class="token operator">*</span><span class="token number">0</span><span class="token punctuation">;</span>

    counter <span class="token operator">=</span> <span class="token number">1</span><span class="token punctuation">;</span>

    <span class="token keyword">while</span> <span class="token operator">~</span><span class="token function">isempty</span><span class="token punctuation">(</span>I<span class="token punctuation">)</span>

      last <span class="token operator">=</span> <span class="token function">length</span><span class="token punctuation">(</span>I<span class="token punctuation">)</span><span class="token punctuation">;</span>

      <span class="token number">i</span> <span class="token operator">=</span> <span class="token function">I</span><span class="token punctuation">(</span>last<span class="token punctuation">)</span><span class="token punctuation">;</span> 

      <span class="token function">pick</span><span class="token punctuation">(</span>counter<span class="token punctuation">)</span> <span class="token operator">=</span> <span class="token number">i</span><span class="token punctuation">;</span><span class="token comment" spellcheck="true">%无条件保留每类得分最高的boundingbox</span>

      counter <span class="token operator">=</span> counter <span class="token operator">+</span> <span class="token number">1</span><span class="token punctuation">;</span>

      xx1 <span class="token operator">=</span> <span class="token function">max</span><span class="token punctuation">(</span><span class="token function">x1</span><span class="token punctuation">(</span><span class="token number">i</span><span class="token punctuation">)</span><span class="token punctuation">,</span> <span class="token function">x1</span><span class="token punctuation">(</span><span class="token function">I</span><span class="token punctuation">(</span><span class="token number">1</span><span class="token operator">:</span>last<span class="token number">-1</span><span class="token punctuation">)</span><span class="token punctuation">)</span><span class="token punctuation">)</span><span class="token punctuation">;</span>

      yy1 <span class="token operator">=</span> <span class="token function">max</span><span class="token punctuation">(</span><span class="token function">y1</span><span class="token punctuation">(</span><span class="token number">i</span><span class="token punctuation">)</span><span class="token punctuation">,</span> <span class="token function">y1</span><span class="token punctuation">(</span><span class="token function">I</span><span class="token punctuation">(</span><span class="token number">1</span><span class="token operator">:</span>last<span class="token number">-1</span><span class="token punctuation">)</span><span class="token punctuation">)</span><span class="token punctuation">)</span><span class="token punctuation">;</span>

      xx2 <span class="token operator">=</span> <span class="token function">min</span><span class="token punctuation">(</span><span class="token function">x2</span><span class="token punctuation">(</span><span class="token number">i</span><span class="token punctuation">)</span><span class="token punctuation">,</span> <span class="token function">x2</span><span class="token punctuation">(</span><span class="token function">I</span><span class="token punctuation">(</span><span class="token number">1</span><span class="token operator">:</span>last<span class="token number">-1</span><span class="token punctuation">)</span><span class="token punctuation">)</span><span class="token punctuation">)</span><span class="token punctuation">;</span>

      yy2 <span class="token operator">=</span> <span class="token function">min</span><span class="token punctuation">(</span><span class="token function">y2</span><span class="token punctuation">(</span><span class="token number">i</span><span class="token punctuation">)</span><span class="token punctuation">,</span> <span class="token function">y2</span><span class="token punctuation">(</span><span class="token function">I</span><span class="token punctuation">(</span><span class="token number">1</span><span class="token operator">:</span>last<span class="token number">-1</span><span class="token punctuation">)</span><span class="token punctuation">)</span><span class="token punctuation">)</span><span class="token punctuation">;</span>

      w <span class="token operator">=</span> <span class="token function">max</span><span class="token punctuation">(</span><span class="token number">0.0</span><span class="token punctuation">,</span> xx2<span class="token operator">-</span>xx1<span class="token operator">+</span><span class="token number">1</span><span class="token punctuation">)</span><span class="token punctuation">;</span>

      h <span class="token operator">=</span> <span class="token function">max</span><span class="token punctuation">(</span><span class="token number">0.0</span><span class="token punctuation">,</span> yy2<span class="token operator">-</span>yy1<span class="token operator">+</span><span class="token number">1</span><span class="token punctuation">)</span><span class="token punctuation">;</span>

      inter <span class="token operator">=</span> w<span class="token operator">.*</span>h<span class="token punctuation">;</span>

      o <span class="token operator">=</span> inter <span class="token operator">./</span> <span class="token punctuation">(</span><span class="token function">area</span><span class="token punctuation">(</span><span class="token number">i</span><span class="token punctuation">)</span> <span class="token operator">+</span> <span class="token function">area</span><span class="token punctuation">(</span><span class="token function">I</span><span class="token punctuation">(</span><span class="token number">1</span><span class="token operator">:</span>last<span class="token number">-1</span><span class="token punctuation">)</span><span class="token punctuation">)</span> <span class="token operator">-</span> inter<span class="token punctuation">)</span><span class="token punctuation">;</span><span class="token comment" spellcheck="true">%计算得分最高的那个boundingbox和其余的boundingbox的交集面积</span>

      I <span class="token operator">=</span> <span class="token function">I</span><span class="token punctuation">(</span>o<span class="token operator">&lt;=</span>overlap<span class="token punctuation">)</span><span class="token punctuation">;</span><span class="token comment" spellcheck="true">%保留交集小于一定阈值的boundingbox</span>

    <span class="token keyword">end</span>

    pick <span class="token operator">=</span> <span class="token function">pick</span><span class="token punctuation">(</span><span class="token number">1</span><span class="token operator">:</span><span class="token punctuation">(</span>counter<span class="token number">-1</span><span class="token punctuation">)</span><span class="token punctuation">)</span><span class="token punctuation">;</span>

    picks<span class="token punctuation">{</span>iS<span class="token number">-4</span><span class="token punctuation">}</span> <span class="token operator">=</span> pick<span class="token punctuation">;</span><span class="token comment" spellcheck="true">%保留每一类的boundingbox</span>

<span class="token keyword">end</span>

</code></pre>

