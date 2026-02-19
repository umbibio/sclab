# Product Guidelines - SCLab

## Prose Style & Voice
- **Accessible and Friendly**: SCLab uses welcoming, easy-to-understand language. While maintaining scientific accuracy, we avoid overly dense technical jargon where simpler explanations suffice. The goal is to lower the barrier for biologists while remaining a serious tool for experts.
- **Supportive Tone**: Labels and descriptions should guide the user, making the complex world of single-cell analysis feel manageable.

## Brand Messaging & Tone
- **Professional and Trustworthy**: We emphasize the reliability and scientific rigor of the underlying methods (scanpy, etc.). Users should feel confident that SCLab's interactive results are as valid as those produced via scripts.
- **Community-Driven and Open**: SCLab is a collaborative effort. Documentation and UI should reflect an open-source spirit, encouraging exploration and contribution.

## Visual Identity & Design Principles
- **Jupyter Consistency**: The SCLab interface is designed to feel like a native extension of the Jupyter ecosystem. We use color palettes, fonts, and spacing that align with the standard JupyterLab and Notebook aesthetics.
- **Data-First Layout**: The primary focus is always on the data. Data visualizations are given priority in the layout, with control panels and tables positioned to support the analysis without overwhelming the plots.

## Interactivity & User Guidance
- **Contextual Assistance**: Help is provided when and where it's needed. We utilize tooltips, hover states, and context-sensitive help buttons to explain parameters and features without cluttering the main interface.
- **Guided Exploration**: Users are encouraged to explore their data, with the UI providing subtle cues and assistance to ensure they stay on a productive analytical path.

## Error Handling & Stability
- **Graceful Degradation**: When an error occurs (e.g., a failed clustering run), SCLab provides a clear, actionable error message. The rest of the dashboard remains functional, allowing the user to adjust parameters or try a different approach without losing their current state.
- **State Resilience**: We strive to maintain the integrity of the underlying `AnnData` object, ensuring that even if an interactive step fails, the data remains safe and consistent.
