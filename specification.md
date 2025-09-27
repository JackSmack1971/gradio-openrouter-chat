# README Update Specification

## Overview
Update the README.md file to accurately reflect the current state of the OpenRouter × Gradio Chat Interface codebase, ensuring all features, architecture, and setup instructions are current and correct.

## Functional Requirements

### 1. Python Version Accuracy
- **Requirement**: Update all references from Python 3.10+ to Python 3.12
- **Rationale**: Dockerfile uses `python:3.12-slim` as base image
- **Locations**: Tech Stack section, Installation prerequisites, any version-specific notes

### 2. Conversation Management System Documentation
- **Requirement**: Document the complete conversation management system including:
  - Persistent conversation storage in `conversations.json`
  - Sidebar UI for conversation management
  - New conversation creation
  - Conversation deletion
  - Conversation switching
  - Automatic conversation saving
- **Rationale**: Current README only mentions export/import JSON but not the full conversation system

### 3. Updated UI Architecture
- **Requirement**: Update architecture diagrams and descriptions to reflect:
  - Gradio Blocks layout with sidebar
  - Responsive design with main chat area and sidebar
  - Conversation list dropdown
  - Control buttons (New, Delete, Export, Import)
- **Rationale**: Current README shows old ChatInterface-only architecture without sidebar

### 4. Project Structure Updates
- **Requirement**: Update project structure to include:
  - `conversations.json` - central conversation storage
  - `.data/` directory contents:
    - `usage.csv` - analytics data
    - `chat_*.json` - exported conversations
- **Rationale**: Structure has evolved beyond initial simple export/import

### 5. Feature Completeness
- **Requirement**: Ensure all implemented features are documented:
  - Conversation persistence across sessions
  - Real-time conversation switching
  - Sidebar-based navigation
  - Enhanced export/import with conversation management
  - Automatic conversation titling
  - UUID-based conversation identification
- **Rationale**: Several features exist in code but are not documented

### 6. Remove Non-Existent References
- **Requirement**: Remove or update references to:
  - Non-existent deployment sections (README mentions deployment section that doesn't exist)
  - Outdated architecture diagrams
  - Incorrect feature descriptions
- **Rationale**: Maintain accuracy and avoid confusion

### 7. Configuration Documentation
- **Requirement**: Ensure all environment variables and settings are accurately documented
- **Rationale**: Config.py has additional settings that may not be fully covered

### 8. Architecture Diagram Updates
- **Requirement**: Update Mermaid diagrams to reflect:
  - Conversation management components
  - Sidebar UI elements
  - Persistent storage layer
- **Rationale**: Current diagrams don't show conversation system

## Non-Functional Requirements

### 1. Documentation Quality
- **Requirement**: Maintain consistent formatting and style
- **Rationale**: Professional presentation

### 2. Accuracy Verification
- **Requirement**: All code references and evidence links must point to existing code
- **Rationale**: Prevent broken links and confusion

### 3. Completeness
- **Requirement**: No implemented features should be undocumented
- **Rationale**: Users need complete feature awareness

## Scope Boundaries

### In Scope
- README.md content updates
- Architecture diagram modifications
- Feature documentation additions
- Configuration table updates
- Project structure documentation

### Out of Scope
- Code modifications
- New feature implementation
- Testing code changes
- Deployment infrastructure changes

## Dependencies
- Current codebase state (main.py, config.py, utils.py, Dockerfile)
- Existing README.md as baseline
- Gradio and OpenRouter API documentation for accuracy

## Success Criteria
- README accurately reflects all current codebase features
- All code evidence links are valid
- Python version is correctly stated as 3.12
- Conversation management system is fully documented
- Architecture diagrams match current implementation
- No references to non-existent sections or features