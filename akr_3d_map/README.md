## 📋 Summary: What We Built

### **Starting Problem**

- You had two similar classes (`Cartesian`, `LTRMLat`) with ~80% duplicate code
- Methods like `create_grid()`, `assign_bin_indices()`, `add_observation_time()` were repeated

### **Solution: Base Class + Mixins Pattern**

```
Observation time (mixin)
              ↓
         AKRGrid (ABC) ← Base functionality
              ↓
      ┌───────┴───────┐
      ↓               ↓
  Cartesian      LTRMLat
```

---

## 🎯 What Each Part Does

### **1. Base Class (`AKRGrid`)**

```python
class AKRGrid(ABC, ResidenceTimeCalculator):
```

- Owns generic attributes: `grid`, `N_DIMENSIONS`
- Implements common methods: `create_grid()`, `assign_bin_indices()`, `_validate_and_get_grid()`
- Defines abstract methods children must implement: `_get_dimension_names()`, `_get_range_attrs()`
- **Inherits from mixins** to get specialized functionality

### **2. Mixins (e.g., `ResidenceTimeCalculator`)**

```python
class ResidenceTimeCalculator:
    def _add_time_intervals(self, df): ...
    def add_observation_time(self, df, coord_colnames): ...
```

- Focused on **one responsibility** (residence time calculations)
- No state of their own - just methods
- Can be reused in other classes if needed

### **3. Child Classes (`Cartesian`, `LTRMLat`)**

```python
class Cartesian(AKRGrid):
    x_range = (-15, 15)  # Natural attributes
    
    def _get_dimension_names(self):
        return ("x", "y", "z")  # Mapping to base class
```

- Define **natural attributes** (`x_range`, `lt_range`, etc.)
- Implement **abstract methods** (provide mappings)
- **Automatically inherit** all base class + mixin methods

---

## 🔄 How Behavior Changes When Adding Mixins

### **Current State: One Mixin**

```python
class AKRGrid(ABC, ResidenceTimeCalculator):
    # Has: create_grid(), assign_bin_indices()
    # Inherited from ResidenceTimeCalculator: add_observation_time()
```

```python
cart = Cartesian()
cart.create_grid()                        # ← From AKRGrid
cart.add_observation_time(df, cols)       # ← From ResidenceTimeCalculator
```

### **Future: Adding More Mixins**

```python
class AKRGrid(ABC, ResidenceTimeCalculator, GridPlotter, StatisticsCalculator):
    # Now has methods from ALL mixins!
```

```python
cart = Cartesian()
cart.create_grid()                        # ← From AKRGrid
cart.add_observation_time(df, cols)       # ← From ResidenceTimeCalculator
cart.plot_3d(variable="observation_time") # ← From GridPlotter (new!)
cart.calculate_statistics()               # ← From StatisticsCalculator (new!)
```

---

## 📊 Impact of Adding Mixins

| When You Add a Mixin | What Happens |
|----------------------|--------------|
| Add to `AKRGrid` parent | **All children get it automatically** |
| No changes to `Cartesian` or `LTRMLat` | ✅ They just work with new methods |
| Write mixin once | 📦 Both coordinate systems can use it |
| Import mixin in base class | 🔄 Single point of integration |

### **Example: Adding a Plotting Mixin**

```python
# 1. Create mixin (once)
class GridPlotter:
    def plot_3d(self, variable, path): ...
    def plot_2d_slice(self, dimension, index): ...
    def export_vtk(self, filename): ...

# 2. Add to base class (one line)
class AKRGrid(ABC, ResidenceTimeCalculator, GridPlotter):
    pass

# 3. ALL children get it automatically
cart = Cartesian()
cart.plot_3d()        # ✅ Works!

lt_grid = LTRMLat()
lt_grid.plot_3d()     # ✅ Works!
```

---

## 🎨 Import Strategy We Set Up

### **Before (confusing)**

```python
from ..base_class_v2 import AKRGrid  # Relative imports
from ..variables import burst_id_colname
```

### **After (clean)**

```python
from scripts.base_class_v2 import AKRGrid  # Absolute imports
from scripts.variables import burst_id_colname
```

**How we did it:**

1. Added `[build-system]` to `pyproject.toml`
2. Added `[tool.setuptools.packages.find]` to specify `scripts` package
3. Ran `uv pip install -e .` to install as editable package
4. Now `scripts` is importable from anywhere

---

## ⚡ Key Benefits

### **1. DRY (Don't Repeat Yourself)**

- Write `add_observation_time()` **once** in mixin
- Both `Cartesian` and `LTRMLat` use it automatically

### **2. Separation of Concerns**

```
AKRGrid:              Core grid logic
ResidenceTimeCalculator: Time calculations
GridPlotter:          Visualization (future)
StatisticsCalculator: Analysis (future)
```

### **3. Easy to Extend**

```python
# To add new coordinate system:
class Spherical(AKRGrid):  # ← Inherits everything!
    r_range = (0, 20)
    theta_range = (0, 2*pi)
    phi_range = (0, pi)
    
    def _get_dimension_names(self):
        return ("r", "theta", "phi")
```

### **4. Type Safety**

```python
def _validate_and_get_grid(self) -> xr.Dataset:
    # Returns typed grid - no assert needed!
```

---

## 🚨 What To Watch Out For

### **1. Method Name Conflicts**

```python
# ❌ Bad: Two mixins with same method name
class MixinA:
    def process(self): ...

class MixinB:
    def process(self): ...  # Conflict!

# Which one gets used? (First in inheritance order)
class AKRGrid(ABC, MixinA, MixinB):  # MixinA.process() wins
```

**Solution:** Use descriptive names: `process_residence_time()`, `process_statistics()`

### **2. Mixin Order Matters**

```python
class AKRGrid(ABC, Mixin1, Mixin2):  # Mixin1 takes precedence
class AKRGrid(ABC, Mixin2, Mixin1):  # Mixin2 takes precedence
```

**Solution:** Put most specific mixins first, ABC first

### **3. Mixin Dependencies**

```python
# Mixin expects certain methods to exist
class ResidenceTimeCalculator:
    def add_observation_time(self):
        grid = self._validate_and_get_grid()  # ← Expects this to exist!
```

**Solution:** Document what mixins expect in docstrings

---

## 📝 Practical Workflow Going Forward

### **Adding New Functionality:**

1. **Create mixin file:**

   ```python
   # scripts/mixin/statistics.py
   class StatisticsCalculator:
       def calculate_mean(self): ...
       def calculate_std(self): ...
   ```

2. **Import and add to base:**

   ```python
   # scripts/base_class_v2.py
   from scripts.mixin.statistics import StatisticsCalculator
   
   class AKRGrid(ABC, ResidenceTimeCalculator, StatisticsCalculator):
       pass
   ```

3. **Use in children automatically:**

   ```python
   cart = Cartesian()
   cart.calculate_mean()  # ✅ Just works!
   ```

---

## 🎯 Bottom Line

**What changed:**

- ✅ Eliminated 80% code duplication
- ✅ Organized code by responsibility (mixins)
- ✅ Made imports clean and absolute
- ✅ Type-safe validation pattern

**How behavior changes:**

- 🔄 Adding mixin = all children get new methods instantly
- 📦 One place to add functionality (base class)
- 🚀 Easy to create new coordinate systems

**Mental model:**

```
Base Class = Core functionality + Container for mixins
Mixins = Specialized capabilities (plug and play)
Children = Coordinate-specific implementations
```

**You now have a scalable, maintainable architecture!** 🎉
