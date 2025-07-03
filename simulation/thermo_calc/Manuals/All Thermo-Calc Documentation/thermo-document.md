> [!PDF|note] [[Thermo-calc 中文手册.pdf#page=8&selection=142,0,156,31&color=note|Thermo-calc 中文手册, p.8]]
> > 体系、组元、相、组成、物种（System, component, phases, constituents and species） 热力学中，总是有一个与环境交换物质、热和功的封闭的或开放的体系，热力学体系由组元和相组成表现为均匀的（均质的）或不均匀的（异质的）状态。组分是体系广义的实体，有时所称的组分是强调这样的事实，一个组分由一个具有某些特征化热力学性质的唯一名称，这些热力学性质为量、活度或化学势。平衡时，整个体系中组分的活度和化学势为常数。在一个体系中，物质将总是出现在一个或多个稳定或介稳定相中（体系均质部分），在一定体积中， 同种相经常出现在很多分开的地方，如空气中粉尘颗粒。均质意味着体系在成分、温度和压力方面是均匀的，并且各处具有相同结构。相比之下，异质体系至少有两相组成。
> 
> > [!PDF|] [[Thermo-Calc_Documentation-Set.pdf#page=55&selection=32,0,76,3|Thermo-Calc_Documentation-Set, p.55]]
> > License type Installation Type Windows macOS Linux SUNLL Standalone Yes Yes Yes NWL Consolidated network Yes No Yes Distributed network Yes No Yes License installations on a server network Yes Yes Yes
> 许可证类型
>

### 1. 系（System）

在热力学中，**系**（System）指的是我们研究的特定部分，它与外界环境（外部世界）通过界面相隔。系的边界可以是实际的（如容器壁）或假想的（如理想化的边界）。系可以分为三种类型：

- **封闭体系**：物质不能进出，但是可以与环境交换能量（热和功）。
    
- **开放体系**：物质和能量都可以与环境交换。
    
- **孤立体系**：既不能与外界交换物质，也不能交换能量，完全封闭。
    

例如，在研究气体的膨胀时，气体本身就可以看作是一个体系，而它与外界的接触界面决定了它是封闭的还是开放的。

### 2. 组元（Component）

**组元**（Component）是组成热力学体系的独立化学物质。在热力学中，一个组元是指一个化学物种，或可以通过化学反应来描述的物质组成部分。组元是描述体系的基本单位。例如，在水与空气的体系中，水和空气分别是两个组元。

组元的数量通常是热力学模型中所涉及的物质种类的最小集合。它们通常通过化学方程式来描述其物质转化。例如，如果我们研究的是金属和其氧化物的反应，那么金属和氧化物就是体系中的两个组元。

### 3. 相（Phase）

在热力学中，**相**（Phase）指的是在一定的条件下，具有均匀物理和化学性质的物质集合。物质在不同的条件下可能呈现出不同的相。常见的相有：

- **固相**：物质的分子或原子紧密排列，几乎没有自由运动。
    
- **液相**：物质分子或原子之间的相互作用较弱，可以自由流动。
    
- **气相**：物质的分子或原子之间的相互作用几乎可以忽略不计，分子自由运动，充满整个容器。

在热力学中，**相**（Phase）指的是在一定的条件下（如温度、压力、组成等），具有一致物理和化学性质的物质集合。简单来说，**相**就是指物质的不同状态或形态，它们在特定的条件下表现出一致的性质。

#### 更通俗的解释：

1. **物质状态**：物质可以存在于不同的形态，比如固态、液态、气态等。每一种形态都叫做一种相。例如，水可以是冰（固态）、液态水（液态）或者水蒸气（气态），这些都叫做不同的相。
    
2. **相的均匀性**：在一个相内，物质的物理性质（如密度、温度、压强）和化学性质（如成分）是均匀的，即在该相的任何部分，它们都是一样的。
    
3. **相的变化**：物质在不同的温度和压力条件下可能会发生相变。例如，水在0°C以下是冰（固态相），在100°C时是水蒸气（气态相）。这些状态的变化称为相变。
    

#### 举个例子：

- **水**：在0°C以下，水是固态的（冰），在0°C到100°C之间是液态的（液体水），而在100°C以上是气态的（水蒸气）。这三种状态分别是不同的相。
    

#### 总结：

相是指物质在某些条件下呈现的一个“统一的状态”，在这个状态下，物质的性质是均匀的。不同的相之间可以通过改变温度或压力等条件而发生转换。


一个体系可以包含多个相，这些相可以是均质的（如纯水），也可以是异质的（如水和冰的混合物）。

### 4. 组成（Composition）

**组成**（Composition）指的是体系中各个组元的相对比例。在热力学中，组成通常通过化学计量比、摩尔分数、质量分数等来表示。例如，水的组成可以表示为氢和氧的比例。

组成对于描述相的行为非常重要，因为不同的组成可以决定体系的物理性质，如熔点、沸点等。比如，海水的组成与纯水不同，含有盐分，这会影响其熔点和沸点。

### 5. 物种（Species）

**物种**（Species）通常是指具体的化学物质，特别是指化学反应中的反应物或生成物。在热力学中，物种可能指一个组元，但也可以指单独的化学分子或离子。例如，在水的电解过程中，氢气和氧气分别是生成的物种，而水则是原始物种。

### 体系的均质性与异质性

- **均质体系**（Homogeneous system）指的是在该体系的每一个部分中，物质的成分、温度和压力是均匀分布的。在均质体系中，不存在不同的相。例如，在一个完全混合的气体中，气体分子的分布是均匀的。
    
- **异质体系**（Heterogeneous system）指的是由两种或多种不同的相组成的体系。例如，水和冰的混合物就是一个异质体系，因为它包含了液态水和固态冰两个不同的相。
    

### 组成与相的关系

每个相都有其特定的**组成**。例如，在水的体系中，冰的组成是H₂O固态，水的组成是H₂O液态。当水处于气相时，它的组成仍然是H₂O，但它的物理状态不同。组成不仅影响体系的物理和化学性质，还与热力学平衡密切相关。

### 组成和化学势

在热力学平衡中，组成的变化会影响**化学势**。化学势是描述物质在体系中“倾向于进入”或“离开”某一相的驱动力量。平衡时，整个体系中的各组元的活度和化学势是常数，并且在不同相之间的化学势相等。这个平衡条件可以通过热力学方程来描述，例如吉布斯自由能最小化原则。

### 总结

- **系**是热力学研究的对象，可以是封闭的、开放的或孤立的。
    
- **组元**是组成系的独立物质种类。
    
- **相**是指在某些条件下具有相同物理和化学性质的物质部分。
    
- **组成**描述的是体系中组元的比例。
    
- **物种**指的是特定的化学物质，常用于描述化学反应中的各物质。


### **均质体系和异质体系的对比**

| 特性          | 均质体系                 | 异质体系                     |
| ----------- | -------------------- | ------------------------ |
| **组成分布**    | 组成均匀，所有位置的物质组成相同。    | 组成不均匀，不同区域的物质组成和性质不同。    |
| **物理和化学性质** | 物理和化学性质一致。           | 物理和化学性质不同，可能有多个相。        |
| **相的数目**    | 只有一个相，物质在体系内是均匀分布的。  | 至少有两个相，存在明确的界面。          |
| **示例**      | 纯气体、溶液、单一相物质（如水、氮气）。 | 油水混合物、冰水混合物、气泡水、多相金属或合金。 |
| **状态**      | 状态在各部分相同。            | 状态在不同部分可能有所不同。           |
> [!PDF|yellow] [[Thermo-calc 中文手册.pdf#page=9&selection=60,0,95,16&color=yellow|Thermo-calc 中文手册, p.9]]
> > 为了近似表达一相中一个带电物质的化学计量比，将电子用作一特定组元，记为/-或 ZE，通常为相组成的一部分。Thermo-Calc 软件包将气体、液体和固相中的带电组成记为/-，将水溶液带电组成记为 ZE，。与这种特定指派相对应，带负电荷组元的化学计量比可表示为 H1O2/-或 H1O2ZE+1，带正电荷组元为 FE1/+2 或 FE1ZE-2。模型中使用的其它特定组元是空位，总是记为 VA。空位用作有空的位置的亚点阵的组成，空位的化学势总是设定为零。
> 
> > [!PDF|yellow] [[Thermo-calc 中文手册.pdf#page=9&selection=97,0,114,8&color=yellow|Thermo-calc 中文手册, p.9]]
> > 空位和电子（气体、液体和固相中为/-，溶液中为 ZE）也作为数据库中特定元素的定义，在含水异质反应体系情况下，ZE（不是/-也不是 VA）也被看作 Thermo-Calc 软件中体系组元。

# Thermo-calc console mode
> [!PDF|yellow] [[Thermo-calc 中文手册.pdf#page=25&selection=223,0,250,3&color=yellow|Thermo-calc 中文手册, p.25]]
> > 普通结构总是可是从菜单中选择命令和从提示框中输入命令。键入问题提示号“？”或在一定模块中给出命令 HELP 获得菜单形式的命令。通常以条命令为用下划线连接起来的几个单词，如 LIST_EQUILIBRIUM。因此通过读取菜单容易理解命令执行的内容。键入跟在 HELP 后命令可得到更广泛的解释。若菜单较长，可尽列出命令的一部分，如以 LIST 开头的所有命令只键入 HELP LIST 即可。
> 
> > [!PDF|yellow] [[Thermo-calc 中文手册.pdf#page=26&selection=17,0,78,14&color=yellow|Thermo-calc 中文手册, p.26]]
> > 命令行输入中几乎所有符号都可用小写字母。因此用户可选择易于记忆的自己的缩写，这个命令行缩写的规则可用于包括相名和其它主题的几乎所有各类输入。下面给出一些实例： 正常命令缩写命令 CALCULATION_EQUILIBRIUM c-e CALCULATION_ALL_EQUILIBRIUM c-a LIST_EQUILIBRIUM l-e LIST_INITIAL_EQUILIBRIUM li-i-e LOAD_INITIAL_EQUILIBRIUM lo-i-e LIST_PHASE_DATA CBCC l-p-d cb LIST_PHASE_DATA CEMENTITE l-p-d ce SET_ALL_START_VALUES s-a-s, or s-al SET_AXIS_VARIABLE 1 X(FCC,FE)0 0.89 0.025 s-a-v1 x(f,fe) 0 .89 .25 SET_START_CONSTITUENT s-s-c SET_START_VALUE s-s-v SET_AXIS_PLOT_STATUS s-a-p SET_AXIS_TEXT_STATUS s-a-t-s, or s-a-te SET_AXIS_TYPE s-a-ty SET_OPTIMIZING_CONDITION s-o-c SET_OPTIMIZING_VARIABLE s-o-v SET_OUTPUT_LEVEL s-o-l, or s-ou
> 
> > [!PDF|yellow] [[Thermo-calc 中文手册.pdf#page=26&selection=86,0,94,39&color=yellow|Thermo-calc 中文手册, p.26]]
> > 系统记住最后 20 个命令，通过键入两个感叹号“！！”可列出。通过键入一个“！”和前面已渐入的命令数就可再次执行同一命令。通过键入“！？”将给出过去使用过的工具的全面解释。
> 
> > [!PDF|yellow] [[Thermo-calc 中文手册.pdf#page=28&selection=238,0,245,14&color=yellow|Thermo-calc 中文手册, p.28]]
> > 所有基本模块都是基本命令行并要求研究时化学势的全面理解是执行模块的基础。一下简要介绍基本模块的功能。表 3-5 Thermo-Calc 软件包中的模块和应用编程界面

![[Pasted image 20250327095535.png]]![[Pasted image 20250327095552.png]]

### 一般操作流程：

1. **定义系统**：在计算开始之前，您需要在 `DATA` 模块中定义系统，并从热力学数据库（如 `.TDB` 文件）中检索所需的数据。
    
2. **选择相和成分**：定义好系统后，选择合适的相和成分。通常在 `POLY` 模块中进行。
    
3. **执行计算**：在 `POLY` 模块中运行所选系统的热力学计算。计算完成后，您可以在 `POST` 模块中查看计算结果并进行图形化展示。
    
4. **结果分析与可视化**：计算结果可以通过 `POST` 模块进行处理和可视化，如生成相图、性质图等。
    

### 控制台模式流程：

1. **切换至控制台模式**：启动 Thermo-Calc 后，您可以通过点击工具栏上的“Switch to Console Mode”按钮来切换到控制台模式 。
    
2. **使用控制台输入命令**：控制台模式下，您通过命令行输入命令来操作系统和执行计算。例如，您可以在 `SYS` 模块中定义系统，在 `DATA` 模块中加载热力学数据，接着使用 `POLY` 模块进行计算 。
    
3. **查看和处理结果**：计算后，结果会显示在控制台的结果窗口中，可以通过命令查看图表和表格数据 。
    
4. **命令历史和日志**：控制台模式支持命令历史功能，您可以通过 `!!` 或 `!<text>` 来回顾和重复之前执行的命令 。
    

### 控制台模式的常见命令：

- **定义系统**：使用 `DEFINE_MATERIAL` 命令定义材料，`DEFINE_DIAGRAM` 命令生成图表。
    
- **计算平衡**：使用 `EQUILIBRIUM` 命令进行热力学计算。
    
- **结果可视化**：在 `POST` 模块中，可以使用 `PLOT_DIAGRAM` 命令生成图形，查看计算结果 。
    

控制台模式的操作主要通过命令行完成，适合进行批量处理和自动化脚本执行。


# SYS 模块
### 1. **ABOUT**

- **用途**：显示 Thermo-Calc 软件的开发历史、所有者以及版本信息。
    
- **用法**：输入 `ABOUT`，将显示基本信息。
    

### 2. **BACK**

- **用途**：返回到最近访问的模块。
    
- **用法**：此命令通常在 `POST` 模块使用，返回到 `TAB` 或 `POLY` 模块。
    


### 3. **CLOSE_FILE**

- **用途**：关闭打开的文件。
    
- **用法**：输入文件的单元编号（通过 `OPEN_FILE` 命令打开文件时获取）。


### 4. **DISPLAY_LICENSE_INFO**

- **用途**：显示当前安装的 Thermo-Calc 软件的许可证信息，包括许可证状态、功能等。
    
- **用法**：输入 `DISPLAY_LICENSE_INFO`，显示详细的许可证信息。
    

### 5. **EXIT**

- **用途**：退出当前模块，或退出控制台模式。
    
- **用法**：在任何模块中输入 `EXIT` 即可退出。
    

### 6. **GOTO_MODULE**

- **用途**：跳转到指定的模块。
    
- **用法**：输入模块名称以切换到该模块。
    


### 7. **HELP**

- **用途**：获取帮助信息，列出可用命令。
    
- **用法**：输入 `HELP` 或 `?` 来查看当前模块的命令列表。你还可以使用 `HELP <command>` 来获取特定命令的详细说明。



### 8. **HP_CALCULATOR**

- **用途**：启动一个简单的交互式计算器，使用逆波兰表示法（RPN）。
    
- **用法**：输入 `HP_CALCULATOR` 来进入交互式计算器。


### 9. **INFORMATION**

- **用途**：获取关于当前模块的基本信息。
    
- **用法**：输入 `INFORMATION`，后跟一个主题名称，获取该主题的详细信息。
    


### 10. **MACRO_FILE_OPEN**

- **用途**：打开并执行一个宏文件，宏文件通常包含一系列 Thermo-Calc 命令。
    
- **用法**：输入宏文件的路径，宏文件通常是 `.TCM` 或 `.DCM` 格式。
    

### 11. **OPEN_FILE**

- **用途**：打开一个文件。
    
- **用法**：指定文件路径。
    

### 12. **SET_COMMAND_UNITS**

- **用途**：设置命令的单位，指定计算中使用的单位制（如 SI 单位，英制单位等）。
    
- **用法**：输入 `SET_COMMAND_UNITS` 并选择单位。
    


### 13. **SET_LOG_FILE**

- **用途**：将输入保存到一个日志文件中，通常用于调试或批处理任务。
    
- **用法**：输入日志文件的路径，开始记录命令序列。
    
### 14. **SET_PLOT_ENVIRONMENT**

- **用途**：设置绘图环境，配置图形输出设备和文件。
    
- **用法**：输入图形设备编号和文件名，设置默认绘图设备。
    

### 15. **STOP_ON_ERROR**

- **用途**：启用错误停止功能，遇到错误时立即停止执行后续命令。
    
- **用法**：输入 `STOP_ON_ERROR` 启用该功能。
    


---

这些是 **System Utilities (SYS)** 模块中常用的一些控制台命令。

接下来是 **POLY 模块** 和其他相关模块中常用的命令，适用于 Thermo-Calc 控制台模式（Console Mode）。我将根据文档逐一解释每个命令的用途、用法和示例。以下是 **POLY 模块** 的命令及其说明。

# **POLY 模块**
### 1. **ADD_INITIAL_EQUILIBRIUM**

- **用途**：添加初始平衡状态，定义开始计算的初始平衡。
    
- **用法**：使用该命令设置初始平衡的元素和相。

### 2. **ADVANCED_OPTIONS**

- **用途**：设置高级选项，允许对计算过程进行更细粒度的控制。
    
- **用法**：输入此命令后，您可以选择更复杂的计算选项。

### 3. **AMEND_STORED_EQUILIBRIA**

- **用途**：修改存储的平衡计算。
    
- **用法**：允许您调整已存储的平衡状态。
### 4. **COMPUTE_EQUILIBRIUM**

- **用途**：执行热力学平衡计算。
    
- **用法**：此命令用于计算当前系统的热力学平衡。

### 5. **CREATE_NEW_EQUILIBRIUM**

- **用途**：创建新的平衡状态，通常在不同的条件下进行多次计算时使用。
    
- **用法**：创建一个新的平衡状态并输入必要的条件。

### 6. **DEFINE_COMPONENTS**

- **用途**：定义系统中的组件，可以是元素或其他物质。
    
- **用法**：在平衡计算中，使用该命令来指定计算所用的组件。
- `DEFINE_COMPONENTS Fe O C`

### 7. **DEFINE_DIAGRAM**

- **用途**：定义图表类型，设置所需绘制的图表。
    
- **用法**：设置图表类型（如相图、性质图等）。
    
- **示例**：
    
    `DEFINE_DIAGRAM Phase Diagram`

### 8. **DELETE_INITIAL_EQUILIBRIUM**

- **用途**：删除初始平衡状态。
    
- **用法**：移除当前设定的初始平衡，适用于重新计算时清理先前的状态。
    
- **示例**：
    
    bash
    
    复制
    
    `DELETE_INITIAL_EQUILIBRIUM`
    

### 9. **DELETE_SYMBOL**

- **用途**：删除已定义的符号。
    
- **用法**：删除常量、变量或其他符号。
    
- **示例**：
    
    
    `DELETE_SYMBOL "X(FE)"`
    

### 10. **ENTER_SYMBOL**

- **用途**：输入新的符号，可以是常量、变量或函数。
    
- **用法**：定义新的符号供计算使用。
    
- **示例**：
    
    bash
    
    复制
    
    `ENTER_SYMBOL CONSTANT X=1.0`
    

### 11. **EQUILIBRIUM_CALCUL**

- **用途**：执行平衡计算，通常在计算热力学平衡时使用。
    
- **用法**：输入命令后，计算指定系统的平衡状态。
    
- **示例**：
    
    bash
    
    复制
    
    `EQUILIBRIUM_CALCUL`
    

### 12. **LIST_EQUILIBRIUM**

- **用途**：列出所有当前计算的平衡状态。
    
- **用法**：显示所有的平衡状态及其信息。
    
- **示例**：
    
    bash
    
    复制
    
    `LIST_EQUILIBRIUM`
    

### 13. **MAP**

- **用途**：绘制系统的相图，展示不同条件下的平衡状态。
    
- **用法**：生成系统的相图或其他类型的图表。
    
- **示例**：
    
    bash
    
    复制
    
    `MAP`
    

### 14. **NEW_COMPOSITION_SET**

- **用途**：创建新的组成集合，定义新的物质组合。
    
- **用法**：用于创建新的元素组合。
    
- **示例**：
    
    bash
    
    复制
    
    `NEW_COMPOSITION_SET "Fe-C"`
    

### 15. **SAVE_WORKSPACES**

- **用途**：保存当前工作空间。
    
- **用法**：保存平衡计算和数据，以便后续使用。
    
- **示例**：
    
    bash
    
    复制
    
    `SAVE_WORKSPACES "FeC_workspace"`





# POST 模块



## 导出数据结果方式

###  一、导出计算结果为图片：

使用以下命令在`POST`模块中绘制并导出图像：

#### （1）`PLOT_DIAGRAM`命令

绘制结果图形。首先设置图像的坐标轴：
```plaintext
SET_DIAGRAM_AXIS X 变量名
SET_DIAGRAM_AXIS Y 变量名
PLOT_DIAGRAM
```

#### （2）保存图像到文件的方法：

Thermo-Calc Console模式下，绘制出图形后，再通过以下命令导出：

- `DUMP_DIAGRAM`：用于将当前显示的图像保存到文件中
```plaintext
DUMP_DIAGRAM
```

程序会提示输入要保存的文件名和路径。

- 交互式保存法（推荐）： 在Console Results窗口中绘制出图像后，右键单击图像，选择`Save as`，可以保存为多种图片格式：`png, jpg, ps, pdf, gif, svg, emf`​Thermo-Calc_Documentati…。

### 二、导出计算结果为数据文件（文本文件、Excel等）：

`POST`模块提供了统一的命令导出数据：

#### （1）`LIST_DATA_TABLE`命令：

此命令用于将计算的结果以表格形式导出：

使用方法： `LIST_DATA_TABLE`

若直接回车则输出到屏幕；若输入文件名，则会保存为Excel电子表格（*.xls）​


#### （2）`MAKE_EXPERIMENTAL_DATAFILE`命令：

将绘制的图形数据以EXP文件格式导出：

使用方法： `MAKE_EXPERIMENTAL_DATAFILE`

程序会提示输入要保存的文件名并默认扩展名为`.EXP`


### **总结（统一流程）：**

在Thermo-Calc Console模式下，标准且统一的导出方法为：

- **图片**：
    
    - 设置轴：`SET_DIAGRAM_AXIS X 变量名`、`SET_DIAGRAM_AXIS Y 变量名`
        
    - 绘制图形：`PLOT_DIAGRAM`
        
    - 保存图形：`DUMP_DIAGRAM` 或 手动在Console Results窗口右键保存。
    - `DUMP_DIAGRAM PNG  G:\pds` 
        
- **数据**：
    
    - 表格数据：`LIST_DATA_TABLE`（输出到屏幕或Excel）
        
	    - 图形数据：`MAKE_EXPERIMENTAL_DATAFILE`（输出到EXP文件）
        

以上命令适用于Thermo-Calc Console模式中的各种计算，包括但不限于平衡计算、扩散计算等，因此可以视作通用的导出命令。



# Graphical Mode (GM) versus Console Mode (CM)

![[Pasted image 20250328215000.png]]
![[Pasted image 20250328215018.png]]
![[Pasted image 20250328215028.png]]
![[Pasted image 20250328215034.png]]> [!PDF|] [[Thermo-Calc_Documentation-Set.pdf#page=1641&selection=24,0,24,12|Thermo-Calc_Documentation-Set, p.1641]]
> > ENTER_SYMBOL
> 
> > [!PDF|] [[Thermo-Calc_Documentation-Set.pdf#page=1641&selection=30,0,31,68|Thermo-Calc_Documentation-Set, p.1641]]
> > Symbols are a useful feature of the POLY and POST modules to define quantities that are convenient. Symbols can be constants, variables, functions or tables
> 
> > [!PDF|] [[Thermo-Calc_Documentation-Set.pdf#page=1641&selection=64,0,65,45|Thermo-Calc_Documentation-Set, p.1641]]
> > Symbols are a useful feature modules to define quantities that are convenient. Symbols can be constants, variables, functions, or tables
> 
> > [!PDF|] [[Thermo-Calc_Documentation-Set.pdf#page=1642&selection=34,0,36,35|Thermo-Calc_Documentation-Set, p.1642]]
> > TABLES are used for listing results from the STEP or MAP commands. A table consists of a list of any number of state variables, functions, or variables. Defined tables can also be used in the POST (post-processor) module
> 
> > [!PDF|] [[Thermo-Calc_Documentation-Set.pdf#page=1642&selection=38,0,44,23|Thermo-Calc_Documentation-Set, p.1642]]
> > There is a special connection between tables and variables. If a variable is used in a table, it is evaluated for each line of the table in the TABULATE command or when the table is used in a plot
> 
> > [!PDF|] [[Thermo-Calc_Documentation-Set.pdf#page=1643&selection=67,0,88,35|Thermo-Calc_Documentation-Set, p.1643]]
> > Variable(s): <Variable(s) in a table> A table consists of a list of state variables or functions. One way to obtain results from a STEP command is through a table. Example: ENTER TABLE K=T,X(LIQ,C),X(LIQ,CR),ACR(C)
> 
> > [!PDF|important] [[Thermo-Calc_Documentation-Set.pdf#page=1643&selection=105,0,128,6&color=important|Thermo-Calc_Documentation-Set, p.1643]]
> > & <Continuation of the definition for the table> The ampersand & means that you can continue to write the table on the new line if one line is not enough for the table. If you finish the table press <Enter> again.
> 
> 

# Fe-C相图

**Fe-C 相图**（铁-碳相图）是用来展示铁和碳在不同温度和碳含量下的相变行为。这个相图描述了不同组成和温度下的钢铁的稳定和不稳定相，包括：

- **固溶体（固溶铁）**
    
- **固相（例如：铁素体、奥氏体）**
    
- **化合物（如水泥石）**
    
- **石墨**等。

![[demo.png]]


Fe-C 相图的作用是展示不同 **碳含量（C）** 和 **温度（T）** 下的 **相行为**。常见的相包括：

- **液相（Liquid）**：熔融状态的铁-碳合金，碳以溶解状态存在。
    
- **铁素体（α-Fe）**：低温下的铁相，含碳量较低。
    
- **奥氏体（γ-Fe）**：高温下的铁相，可以溶解更多碳。
    
- **水泥石（Fe₃C）**：碳在高温下形成的化合物。
    
- **石墨（Graphite）**：碳的固体相，常见于较高碳含量和低温下。
    

在 **Fe-C 相图** 中，你通常会看到以下几个关键区域：

- **液相线**：温度升高时，铁和碳从固态转变为液态。
    
- **共晶点**：此点表示在某一温度下，铁和碳在某个比例下可以同时固化成不同相。
    
- **固相线**：描述不同固体相之间的转变。
    
- **碳含量**：从低碳到高碳，表现不同固溶体和化合物的稳定性。



# 问题 

## 1. STEP calculation?

> [!PDF|] [[Thermo-Calc_Documentation-Set.pdf#page=1645&selection=35,10,35,43|Thermo-Calc_Documentation-Set, p.1645]]
> > but only after a STEP calculation.
> 
> 啥意思? 什么叫 STEP calculation ? 