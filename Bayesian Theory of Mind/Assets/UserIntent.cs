using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class UserIntent : MonoBehaviour
{
    [SerializeField] private GameObject agent1;
	[SerializeField] private GameObject agent2;
    [SerializeField] private GameObject rightEye;
    [SerializeField] private GameObject leftEye;
    [SerializeField] private GameObject head;
    private RayCast rightEyeRayCast;
    private RayCast leftEyeRayCast;
    private RayCast headRayCast;
    private Vector3 prevPosition;
    private Vector3 currPosition;
    private int agentSelector;

    // Start is called before the first frame update
    void Start()
    {
        rightEyeRayCast = rightEye.GetComponent<RayCast>();
        leftEyeRayCast = leftEye.GetComponent<RayCast>();
        headRayCast = head.GetComponent<RayCast>();
        prevPosition = head.transform.position;
        currPosition = head.transform.position;
        agentSelector = 0;
    }

    void FixedUpdate()
    {
        agentSelector = 0;
        prevPosition = currPosition;
        currPosition = head.transform.position;

        // Bit shift the index of the layer (8) to get a bit mask
		int layerMask = 1 << 8;

		// This would cast rays only against colliders in layer 8.
		// But instead we want to collide against everything except layer 8. The ~ operator does this, it inverts a bitmask.
		layerMask = ~layerMask;

		RaycastHit hit;
        if(Physics.Raycast(currPosition, (currPosition - prevPosition), out hit, Mathf.Infinity, layerMask))
		{
            if(hit.collider.tag == "Agent 1")
			{
                agentSelector += 1;
            }
            else if(hit.collider.tag == "Agent 2")
            {
                agentSelector -= 1;
            }
            else
            {
                agentSelector = 0;
            }
        }
    }

    // Update is called once per frame
    void Update()
    {
        agentSelector += rightEyeRayCast.agentSelected + leftEyeRayCast.agentSelected + headRayCast.agentSelected;
        if(agentSelector > 0)
        {
            agent1.GetComponent<Outline>().enabled = true;
            agent2.GetComponent<Outline>().enabled = false;
        }
        else if(agentSelector < 0)
        {
            agent1.GetComponent<Outline>().enabled = false;
            agent2.GetComponent<Outline>().enabled = true;
        }
        else
        {
            agent1.GetComponent<Outline>().enabled = false;
            agent2.GetComponent<Outline>().enabled = false;
        }
    }
}
