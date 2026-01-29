## [Unreleased]

## [0.3.0]

This release focused on polishing the user interface.

### New

- Updated and added new boxes for using [PyKEEN](https://github.com/pykeen/pykeen) graph embeddings.
  [#62](https://github.com/lynxkite/lynxkite-2000/pull/62)
- A UI refresh. Boxes have icons now.
  [#84](https://github.com/lynxkite/lynxkite-2000/pull/84)
  [#112](https://github.com/lynxkite/lynxkite-2000/pull/112)
- A box's type can be changed.
  [#100](https://github.com/lynxkite/lynxkite-2000/pull/100)
  [#102](https://github.com/lynxkite/lynxkite-2000/pull/102)
- A new parameter type that allows choosing a table and its column. [#109](https://github.com/lynxkite/lynxkite-2000/pull/109)
- Tooltip improvements.
  [#104](https://github.com/lynxkite/lynxkite-2000/pull/104)
  [#93](https://github.com/lynxkite/lynxkite-2000/pull/93)
  [#103](https://github.com/lynxkite/lynxkite-2000/pull/103)
- Pan with left click drag, select with Shift drag. [#92](https://github.com/lynxkite/lynxkite-2000/pull/92)
- Box search no longer off screen, zoom on scroll, bottom left buttons removed.
  [#90](https://github.com/lynxkite/lynxkite-2000/pull/90)
- Resizable nicer groups. [#85](https://github.com/lynxkite/lynxkite-2000/pull/85)
- Boxes and groups snap to grid when Shift is held. [#86](https://github.com/lynxkite/lynxkite-2000/pull/86)
- Custom icons, scrolling tables. [#88](https://github.com/lynxkite/lynxkite-2000/pull/88)

### Fixes

- Match `<input>` borders on `<textarea>` and `<select>` elements. [#94](https://github.com/lynxkite/lynxkite-2000/pull/94)
- Stop collapsed boxes from blocking things. [#83](https://github.com/lynxkite/lynxkite-2000/pull/83)
- Fix extent=null issue. [#81](https://github.com/lynxkite/lynxkite-2000/pull/81)
- Fix restoring height after uncollapsing box. [#127](https://github.com/lynxkite/lynxkite-2000/pull/127)
- Fix unintentional text selection when trying to select boxes. [#126](https://github.com/lynxkite/lynxkite-2000/pull/126)
- CRDT fixes. [#116](https://github.com/lynxkite/lynxkite-2000/pull/116)
- Fix dropdowns. [#101](https://github.com/lynxkite/lynxkite-2000/pull/101)

### Internal improvements

- Make "name" a required field for relation definitions. [#95](https://github.com/lynxkite/lynxkite-2000/pull/95)
- Upgrade all JavaScript dependencies. Connect CRDT and ReactFlow ourselves.
  [#89](https://github.com/lynxkite/lynxkite-2000/pull/89)
- Do not install CUDA dependencies in CI. [#91](https://github.com/lynxkite/lynxkite-2000/pull/91)
- Use uv build/publish from top-level directory. [#80](https://github.com/lynxkite/lynxkite-2000/pull/80)
- Bump uv_build version. [#79](https://github.com/lynxkite/lynxkite-2000/pull/79)
- Started a changelog.
