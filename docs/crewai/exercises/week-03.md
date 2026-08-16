# Exercises — Week 3 — Team Collaboration & Communication

Do these after reading [Week 3 — Team Collaboration & Communication](../week-03.md).

## ✍️ Hands-On Exercises

!!! example "Exercise"

    **🎯 Exercise 1: Delegation Workflow**

    Build a multi-agent delegation chain:

    - CEO delegates to Product Manager (define requirements)

    - Product Manager output → Architect (design system)

    - Architect output → Security Lead (add security)

    - All outputs → CEO (final decision)



```python
# Your implementation here!
# Create tasks that flow from CEO → specialist → specialist → back to CEO

print("Your delegation workflow here!")
```

!!! example "Exercise"

    **🎯 Exercise 2: Consensus Mechanism**

    Implement consensus for team decisions:

    - Propose technical approach

    - Each specialist evaluates and votes

    - Aggregate votes and identify objections

    - Discuss objections and re-vote if needed

    - Reach consensus or escalate to CEO



```python
# Your implementation here!
print("Your consensus mechanism here!")
```

## 📝 Week 3 Project: Software Design Review Team

**Build a multi-agent team that reviews software designs.**

### Team Members (5+ Agents):
1. **Product Manager**: Requirements and user needs
2. **Technical Architect**: System design and scalability
3. **Security Lead**: Security and compliance
4. **Performance Expert**: Performance and optimization
5. **DevOps Lead**: Deployment and operations
6. **CEO/Director**: Makes final decisions

### Design Review Process:
1. Product Manager presents requirements
2. Architect proposes design
3. Each specialist reviews and provides feedback
4. CEO aggregates input and makes decision
5. Document final design with rationale

### Test Scenarios:
1. **Simple**: CRUD API for user profiles
2. **Complex**: Real-time collaboration platform
3. **Urgent**: 2-week timeline constraint
4. **Conflicting**: Scalability vs budget vs timeline

```python
# Week 3 Project Starter

# TODO: Create 6 specialized agents (Product, Architect, Security, Performance, DevOps, CEO)
# TODO: Create 5+ tasks with delegation chains
# TODO: Implement consensus/voting mechanism
# TODO: Test with 3 different design scenarios
# TODO: Document team decisions and trade-offs

print("🎯 Your software design review team here!")
```

## 🎓 Key Takeaways

**What you learned this week:**

✅ **Communication Patterns:**
- Direct communication via context
- Shared state repositories
- Hierarchical decision-making

✅ **Delegation:**
- When to delegate vs decide yourself
- Effective delegation instructions
- Feedback and iteration

✅ **Conflict Resolution:**
- Understanding different perspectives
- Finding middle ground
- Clear trade-off documentation

✅ **CEO Pattern:**
- Gather specialist input
- Facilitate discussion
- Make informed decisions

## 🔜 Next Week: Production & Scaling

In Week 4, we'll map the operational concerns around agents:
- Monitoring and observability
- Error handling and recovery
- Scaling strategies
- Production best practices

## 📚 Additional Resources

- [CrewAI Agent Communication](https://docs.crewai.com/)
- [Team Collaboration Patterns](https://en.wikipedia.org/wiki/Team_collaboration)
- [Consensus Algorithms](https://en.wikipedia.org/wiki/Consensus_(computer_science))

---

**🎉 Congratulations on completing Week 3!** You can now build collaborative teams of agents. See you next week! 🚀
