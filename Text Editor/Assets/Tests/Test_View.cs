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

    [TestCase(-0.4855f, 0.4669f, 0, 0)] // Upper left
    [TestCase(-0.4793f, 0.4379f, 0, 0)] // Bottom right
    [TestCase(-0.4830f, 0.4545f, 0, 0)] // Centre
    [TestCase(0.4731f, -0.4159f, 14, 52)] // Upper left
    [TestCase(0.4830f, -0.4470f, 14, 52)] // Bottom right
    [TestCase(0.4781f, -0.4325f, 14, 52)] // Centre
    [TestCase(-0.0075f, 0.0282f, 7, 26)] // Upper left
    [TestCase(0.0025f, 0.0012f, 7, 26)] // Bottom right
    [TestCase(-0.0025f, 0.0157f, 7, 26)] // Centre
    [TestCase(0.4781f, 0.4639f, 0, 52)] // Centre
    [TestCase(-0.4830f, -0.4325f, 14, 0)] // Centre
    [TestCase(-0.2614f, -0.3059f, 12, 12)] // Centre
    [TestCase(0.3287f, 0.2066f, 4, 44)] // Centre
    public void Test_CoordsToInd(float x_coord, float y_coord, int exp_row, int exp_col)
    {
        Vector3 coords = new Vector3(x_coord, y_coord, 0f);
        Indices actual = view.CoordsToInd(coords);
        Assert.AreEqual(exp_row, actual.row, "Incorrect row");
        Assert.AreEqual(exp_col, actual.col, "Incorrect column");
    }

    [TestCase(0, 0 , -0.4830f, 0.4545f)]
    [TestCase(14, 52, 0.4781f, -0.4325f)]
    [TestCase(7, 26, -0.0025f, 0.0157f)]
    [TestCase(0, 52, 0.4781f, 0.4639f)]
    [TestCase(14, 0 , -0.4830f, -0.4325f)]
    [TestCase(12, 12 , -0.2614f, -0.3059f)]
    [TestCase(4, 44, 0.3287f, 0.2066f)]
    public void Test_IndToCoords(int row, int col, float exp_x, float exp_y)
    {
        Vector3 coords = view.IndToCoords(row, col);
        float allowed_delta = 0.015f;
        Assert.AreEqual(exp_x, coords.x, allowed_delta, "X coordinate different");
        Assert.AreEqual(exp_y, coords.y, allowed_delta, "Y coordinate different");
    }

}
