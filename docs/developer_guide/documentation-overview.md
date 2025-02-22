# SysIdentPy Documentation Restructuring Proposal

This document outlines a reorganization of SysIdentPy’s documentation to improve discoverability, reduce friction for beginners, and align with modern documentation standards. The structure will follow four key categories: **Tutorials**, **How-Tos**, **Explanations**, and **Reference Guide**, with additional sections for contributors and real-world examples.

> Acknowledgments: This documentation restructuring draws inspiration from [NumPy’s NEP 44](https://numpy.org/neps/nep-0044-restructuring-numpy-docs.html#nep44), adapting its principles of clarity and logical organization to SysIdentPy’s domain-specific needs in system identification and time series forecasting, while emphasizing tutorials and reproducibility.

## Motivation and Scope

SysIdentPy’s current documentation (like many scientific Python packages) mixes conceptual explanations, code examples, and API references, which can overwhelm new users. By adopting a user-centric structure inspired by [Diátaxis](https://diataxis.fr/), we aim to:

- Separate learning paths for **beginners** (Tutorials) and **practitioners** (How-Tos).
- Improve material for conceptual understanding (Explanations).
- Maintain a clean, searchable **Reference Guide**.
- Highlight SysIdentPy’s features.

A well-organized documentation structure can significantly improve the experience of our community by providing specific resources for different user groups:

- For Beginners: A clear, guided pathway with tutorials and step-by-step instructions helps new users overcome the learning curve.

- For Researchers: Features such as custom basis functions and model configurations can be easily discovered and understood. With clearly defined sections, researchers can quickly locate the information they need to experiment with new methods.

- For Industry/Corporate Users: Benchmarking guides and model comparison examples are readily accessible, making it easier for industry and corporate professionals to evaluate and choose the right tools for their specific needs.

The goal is to structure the documentation to meet the specific needs of these diverse user groups, making the learning process faster and more efficient for everyone in the community.

## Proposed Structure

Here’s an overview of the main sections in the documentation, outlining the purpose and proposed content for each:

- Getting Started
- User Guide
- Developer Guide
- Community & Support
- About

### User Guide

The User Guide section is designed to provide a comprehensive understanding of SysIdentPy, covering essential concepts, practical examples, and advanced features. The proposed structure includes:


#### Tutorials

Audience: New users with minimal system identification experience.

Suggested Content:

<div class="grid cards" markdown>
- :material-book-open-variant: __Absolute Beginner’s Guide__
  Start from scratch with easy-to-follow guides designed for those new to SysIdentPy and NARMAX models.
- :material-application-cog: __Domain-Specific Tutorials__
  Examples and use cases for fields like engineering, healthcare, finance and so on.
</div>

Format: Jupyter Notebooks with narrative explanations and code.


#### How-Tos

Audience: Practitioners solving specific problems.

Suggested Content:

<div class="grid cards" markdown>
- :material-tune: __Model Optimization__
- :material-rocket-launch: __Advanced Customizations__
- :material-chart-box: __Error Analysis__
- :material-repeat: __Reproducibility__
</div>

Format: Short, task-focused markdown files with code snippets.

#### Explanations

Audience: Users seeking rigorous mathematical foundations.

<div class="grid cards" markdown>
- :material-book-open-page-variant: __Companion Book__
  [Nonlinear System Identification and Forecasting: Theory and Practice with SysIdentPy](https://sysidentpy.org/book/0%20-%20Preface/). Offer theoretical context for SysIdentPy’s algorithms through a companion book.
</div>

#### Reference Guide

Audience: Advanced users needing API details.

<div class="grid cards" markdown>
- :material-code-tags: __API Reference__
  Access the complete **SysIdentPy** source code with well-documented modules and methods.
</div>

Format: Auto-generated API docs with cross-linked "See Also" sections.

### Developer Guide

The Developer Guide section aims to provide clear information on the internal structure of SysIdentPy, focusing on implementation details, code examples, and options for customization. The proposed structure includes:

#### How To Contribute

Audience: Maintainers and open-source contributors.

<div class="grid cards" markdown>
- :material-account-plus: __Contributor Guide__
</div>

#### Documentation Guide

Audience: Maintainers and open-source contributors.


<div class="grid cards" markdown>
- :material-book-edit: __Writing a tutorial__
- :material-book-edit: __Creating a how-to guide__
- :material-book-edit: __Creating content for the book__
</div>

### Community & Support

Audience: Individuals of all experience levels, from beginners to experts, with an interest in Python and SysIdentPy.

<div class="grid cards" markdown>
- :material-lifebuoy: __Get Help__
- :material-video: __Workshops__
- :material-book-open-page-variant: __Reading Suggestions__
- :material-forum: __Community Discussions__
</div>
