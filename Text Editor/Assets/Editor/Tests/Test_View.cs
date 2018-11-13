using UnityEngine;
using UnityEngine.TestTools;
using NUnit.Framework;
using System.Collections;

public class TestViewCoordinates
{
    PageView view;

    [SetUp]
    public void Init()
    {
        GameObject page = GameObject.FindGameObjectWithTag("Page");
        view = page.GetComponent<PageView>();
    }

    [TestCase(-0.4801f, 0.471f, 0, 0)] // Upper left
    [TestCase(-0.4667f, 0.446f, 0, 0)] // Bottom right
    [TestCase(-0.4714f, 0.459f, 0, 0)] // Centre
    [TestCase(0.4613f, -0.405f, 14, 52)] // Upper left
    [TestCase(0.4741f, -0.433f, 14, 52)] // Bottom right
    [TestCase(0.4674f, -0.419f, 14, 52)] // Centre
    [TestCase(-0.0092f, 0.0358f, 7, 26)] // Upper left
    [TestCase(0.0047f, 0.004f, 7, 26)] // Bottom right
    [TestCase(-0.0013f, 0.0189f, 7, 26)] // Centre
    [TestCase(0.4669f, 0.4591f, 0, 52)] // Centre
    [TestCase(-0.4723f, -0.421f, 14, 0)] // Centre
    [TestCase(-0.255f, -0.2961f, 12, 12)] // Centre
    [TestCase(0.3223f, 0.2032f, 4, 44)] // Centre
    public void Test_CoordsToInd(float x_coord, float y_coord, int exp_row, int exp_col)
    {
        Vector3 coords = new Vector3(x_coord, y_coord, 0f);
        Indices actual = view.CoordsToInd(coords);
        Assert.AreEqual(exp_row, actual.row, "Incorrect row");
        Assert.AreEqual(exp_col, actual.col, "Incorrect column");
    }

    [TestCase(0, 0 , -0.4714f, 0.459f)]
    [TestCase(14, 52, 0.4674f, -0.419f)]
    [TestCase(7, 26, -0.0013f, 0.0189f)]
    [TestCase(0, 52, 0.4781f, 0.4639f)]
    [TestCase(14, 0 , -0.4723f, -0.421f)]
    [TestCase(12, 12 , -0.255f, -0.2961f)]
    [TestCase(4, 44, 0.3223f, 0.2032f)]
    public void Test_IndToCoords(int row, int col, float exp_x, float exp_y)
    {
        Vector3 coords = view.IndToCoords(row, col);
        float allowed_delta = 0.015f;
        Assert.AreEqual(exp_x, coords.x, allowed_delta, "X coordinate different");
        Assert.AreEqual(exp_y, coords.y, allowed_delta, "Y coordinate different");
    }

}
