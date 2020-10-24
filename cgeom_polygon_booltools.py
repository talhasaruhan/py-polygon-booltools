import random
import math
import numpy as np

np.seterr(all='raise')

# create_samples(DIR_DATASET, DIR_OUTPUT, JSON_DAMAGES, JSON_PARTS, NUM_SAMPLES)

G_FLOAT_EPSILON = 1e-6

# Precompute some values to avoid costly divisions later in the quadratic time algorithm
# input: vertices -> np.array[N, 2], vertices[_, :] = (x, y)
# output: edges -> np.array[N, 4],
# edges[_, :] = (x1, y1, 1/(x2-x1), 1/(y2-y1), m, b)
# where x1, y1 and x2, y2 are the endpoints of the edge and
# m and b constitute the implicit equation : y = m*x + b
# if the edge is vertical following values are used:
# m = INF, b = x where the edge intersects the x axis
def precompute_edge_eqs(vertices):
    def do_computation(x1, y1, x2, y2):
        delta_x = (x2-x1)
        delta_y = (y2-y1)

        zero_deltax = abs(delta_x) < G_FLOAT_EPSILON
        zero_deltay = abs(delta_y) < G_FLOAT_EPSILON

        if zero_deltax and zero_deltay: # same points
            assert False, "Points are the same!"
            return
        elif zero_deltax: # vertical
            inv_deltax = math.inf
            inv_deltay = 1 / delta_y
            m = math.inf
            b = x1
        elif zero_deltay: # horizontal
            inv_deltax = 1 / delta_x
            inv_deltay = math.inf
            m = delta_y / delta_x
            b = y1 - m*x1
        else: # regular case
            inv_deltax = 1 / delta_x
            inv_deltay = 1 / delta_y
            m = delta_y / delta_x
            b = y1 - m*x1

        return inv_deltax, inv_deltay, m, b

    assert len(vertices.shape) == 2 and vertices.shape[1] == 2, \
            "input must be a numpy array of [N, 2], where [_, :] -> (x, y)"

    num_vertices = vertices.shape[0]-1
    edges = np.zeros((num_vertices, 6), dtype = np.float64)

    # We reuse old values from the last iteration, set these up first
    x2, y2 = vertices[0, :]
    for vert_idx in range(num_vertices):
        x1 = x2
        y1 = y2
        x2, y2 = vertices[vert_idx+1, :]
        inv_deltax, inv_deltay, m, b = do_computation(x1, y1, x2, y2)
        edges[vert_idx, :] = [x1, y1, inv_deltax, inv_deltay, m, b]

    return edges

# input: array of precomputed edges np.array[(x1, y1, invdx, invdy, m, b), ...]
# Returns (bool)intersection_found, intersection_coord
# if the lines are parallel, they don't intersect for our purposes,
# even if they're colinear, we ignore that and return False
def find_edge_intersection(edge_1, edge_2):
    x11, y11, invdx1, invdy1, m1, b1 = edge_1
    x21, y21, invdx2, invdy2, m2, b2 = edge_2

    # y11 y22 x11 x22

    # Handle edge cases where edge_1 or edge_2 is vertical.
    # No need to handle horizontal cases since arithmetic operations
    # are cheaper than branches and with m=0 we get the right answer.
    is_edge_1_vertical = math.isinf(m1)
    is_edge_2_vertical = math.isinf(m2)

    if is_edge_1_vertical and is_edge_2_vertical:
        # both vertical, return false
        return False, (0, 0), 0, 0
    if is_edge_1_vertical and not is_edge_2_vertical:
        # edge_1 is vertical, crosses x at b1
        x = b1
        y = m2*x + b2
        t1 = (y-y11)*invdy1
        t2 = (x-x21)*invdx2
    elif not is_edge_1_vertical and is_edge_2_vertical:
        # edge_2 is vertical, crosses x at b2
        x = b2
        y = m1*x + b1
        t1 = (x-x11)*invdx1
        t2 = (y-y21)*invdy2
    else:
        # y = m1*x + b1 = m2*x + b2, x*(m1-m2) = b2-b1
        delta_m = (m1-m2)
        if abs(delta_m) < G_FLOAT_EPSILON: # use epsilon to avoid floating point errors
            # Parallel lines
            return False, (0, 0), 0, 0
        x = (b2-b1) / delta_m
        y = m1*x + b1
        t1 = (x-x11)*invdx1
        t2 = (x-x21)*invdx2

    # Ignore t == 0 and t == 1 cases for now
    if t1 > 0.0 and t1 < 1.0 and t2 > 0.0 and t2 < 1.0:
        return True, (x, y), t1, t2
    else:
        return False, (x, y), t1, t2

# Input: poly: array of vertices defining polygon, point: point to be queried
# Optimized, but equivalent to shooting rays from point in +X direction
# and counting the intersections with polygon edges. This could work with any direction,
# but with +X, we can just check y values to determine if and edge intersects with point.
# Also, we need to determine if the intersection point lays on the correct side of the point (+X).
# Here, instead of calculating the point explicitly and checking x coordinates,
# we multiply all the terms by the division factor to eliminate division altogether.
def is_point_inside_polygon(poly, point):
    num_vertices = len(poly)-1
    winding_number = 0

    # We reuse values from the last iteration,
    # so need to set up these initially
    point_x, point_y = point
    x2, y2 = poly[0, :]
    delta_x2 = x2 - point_x
    delta_y2 = y2 - point_y
    # Since we're shooting rays along X axis, we can just check y values
    # and we can incorporate upwards & downwards edges into the equation
    # by keeping if v(i), v(i+1) are "under" point (v_y < point_x)
    v2_under_pt = delta_y2 <= 0
    for vert_idx in range(num_vertices):
        x1 = x2
        y1 = y2
        delta_x1 = delta_x2
        delta_y1 = delta_y2
        v1_under_pt = v2_under_pt

        x2, y2 = poly[vert_idx+1, :]
        delta_x2 = x2 - point_x
        delta_y2 = y2 - point_y
        v2_under_pt = delta_y2 <= 0

        # is_left = (x1 - px)*(y2 - y1) - (x2 - x1)*(y1 - px)
        #         = (x1 - px)*(y2 - py) - (x2 - px)*(y1 - py)
        # reuse already computed values
        is_left = delta_x1*delta_y2 - delta_y1*delta_x2;

        # if v1_under_pt and not v2_under_pt: # Upwards edge, intersects point
        #     if is_left > 0: winding_number += 1
        # elif v2_under_pt and not v1_under_pt: # Downwards edge, intersects point
        #     if is_left < 0: winding_number -= 1
        # Cast bool to int and do arithmetic, faster than branching
        winding_number += (
                (v1_under_pt & (v2_under_pt ^ 1) & (is_left > 0)).astype(np.int32) -
                (v2_under_pt & (v1_under_pt ^ 1) & (is_left < 0)).astype(np.int32))

    return winding_number != 0

# IF is_debug=False (default):
# Returns list of polygons, where each polygon is another disjoint intersection
# If is_debug is set, who knows what we may be returning,
# check the code. This comment will be outdated soon.
def find_polygon_intersection(poly_1, poly_2, float_coords=False, is_debug = False):
    # n = num of verts = num of edges
    n_1 = len(poly_1)
    n_2 = len(poly_2)
    n_tot = n_1 + n_2

    # Create the initial graphs
    # Vi -> ith vert of poly if i < n
    #    -> (i-(n_1+n_2))th vert of intersection_vertices if i >= n_1+n_2
    # this is so that we don't need to insert same vertex into both graphs
    # and the intersection vertices can be shared in one buffer
    # edge: (prev_vert, next_vert)
    graph_1_edges = {}
    graph_2_edges = {}
    for vert_idx in range(n_1):
        graph_1_edges[vert_idx] = [(vert_idx-1)%n_1, (vert_idx+1)%n_1]
    for vert_idx in range(n_2):
        graph_2_edges[vert_idx] = [(vert_idx-1)%n_2, (vert_idx+1)%n_2]

    # Close the loop
    poly_1.append(poly_1[0])
    poly_2.append(poly_2[0])

    # Convert to np.array
    poly_1 = np.array(poly_1)
    poly_2 = np.array(poly_2)

    # Precompute necessary constants to avoid repeated muls & divs
    edges_1 = precompute_edge_eqs(poly_1)
    edges_2 = precompute_edge_eqs(poly_2)

    # Find vertices of poly_1 that are inside poly_2
    ply1_pts_inside_ply2 = set()
    for pt_idx, point in enumerate(poly_1):
        if is_point_inside_polygon(poly_2, point):
            ply1_pts_inside_ply2.add(pt_idx)

    # Find vertices of poly_2 that are inside poly_1
    ply2_pts_inside_ply1 = set()
    for pt_idx, point in enumerate(poly_2):
        if is_point_inside_polygon(poly_1, point):
            ply2_pts_inside_ply1.add(pt_idx)

    # Edge intersections
    intersection_vertices = []
    # In order to make sure multiple intersections of edge_1 with
    # poly_2 is handled properly, store all the intersections
    # along with "t" parameter (if e1->(p, q), int = p + (q-p)*t)
    # sort by t and then reconstruct the graph.
    # We can do this at the end of every inner loop.
    # However, a similar problem arises if we try to reconstruct graph_2
    # before iterating through all edges in poly_1 that can possibly
    # intersect a given edge in poly_2. So we save those as well
    # and do a similar second pass over graph_2.
    # In order to avoid code duplication, we do graph_1 in the post-pass as well.
    # Also, later on we'll need to know which way from an intersection
    # "goes into" the other polygon. We can compute that during the second pass as well.
    intersection_database_1 = {}
    intersection_database_2 = {}

    # edge_1: edge from poly_1[vert_1_idx] to poly_1[vert_1_idx+1]
    for vert_1_idx in range(n_1):
        # edge_2: edge from poly_2[vert_2_idx] to poly_2[vert_2_idx+1]
        for vert_2_idx in range(n_2):
            edge_1 = edges_1[vert_1_idx, :]
            edge_2 = edges_2[vert_2_idx, :]
            is_intersecting, pt, t1, t2 = find_edge_intersection(edge_1, edge_2)

            if is_intersecting:
                # Add the new intersection into the intersection vertices
                new_vert_idx = n_tot + len(intersection_vertices)
                intersection_vertices.append(pt)

                if vert_1_idx not in intersection_database_1:
                    intersection_database_1[vert_1_idx] = [(t1, new_vert_idx)]
                else:
                    intersection_database_1[vert_1_idx].append((t1, new_vert_idx))
                if vert_2_idx not in intersection_database_2:
                    intersection_database_2[vert_2_idx] = [(t2, new_vert_idx)]
                else:
                    intersection_database_2[vert_2_idx].append((t2, new_vert_idx))

    # Define a submethod to reconstruct graphs, avoid code duplication
    # returns the "direction database" that holds the information of which way
    # "goes into" the other polygon
    def reconstruct_graph(num_vertices, intersection_database, \
            graph_edges, pts_inside_other_ply):
        # Since we haven't modified the graph before this should hold True
        assert num_vertices == len(graph_edges), "?? Problem with reconstruct_graph #verts!"

        direction_db = {}

        for vert_idx, intersections in intersection_database.items():
            # Sort all intersections by increasing t value (default behaviour with tuples)
            intersections.sort()
            intersections.insert(0, (-1, vert_idx))
            intersections.append((-1, (vert_idx+1)%num_vertices))
            num_intersections = len(intersections)

            prev_idx = 0
            current_idx = vert_idx
            for intersection_idx in range(1, num_intersections):
                prev_idx = current_idx
                current_idx = intersections[intersection_idx][1]
                is_sp_inside_otherply = vert_idx in pts_inside_other_ply
                # At each vert, set next of prev and prev of current
                graph_edges[prev_idx][1] = current_idx
                if current_idx in graph_edges:
                    graph_edges[current_idx][0] = prev_idx
                else:
                    graph_edges[current_idx] = [prev_idx, -1]
                if intersection_idx < num_intersections-1:
                    # Assume an edge from vert_idx to vert_idx+1
                    # After sorting the intersections of that edge with the other polygon
                    # by increasing distance to starting point, vert_idx,
                    # given nth intersection int_vert_n (1 based index),
                    # vector V from int_vert_n to vert_idx+1
                    # can either go into or out of the other polygon.
                    # Which can be expressed as whether or not
                    # the point P very next to int_vert_n along the direction of V
                    # is inside the other polygon or not.
                    #   P = (pos of int_vert_n) + V*epsilon
                    # One can see that if n%2 == 0, P is on the same side of the other
                    # polygon as the starting point (vert_idx) is.
                    direction_db[current_idx] = (intersection_idx % 2 != is_sp_inside_otherply)

        return direction_db

    int_direction_db_1 = reconstruct_graph(n_1, intersection_database_1, \
            graph_1_edges, ply1_pts_inside_ply2)
    int_direction_db_2 = reconstruct_graph(n_2, intersection_database_2, \
            graph_2_edges, ply2_pts_inside_ply1)

    # Now begin traversing!
    # There can be multiple disjoint regions where polygons intersect.
    # However in all cases all the intersection vertices must be in exactly one region.
    # (exactly one because we don't allow overlapping vertices, so every region
    # must be completely disjoint. (=== set of manifold boundaries))
    n_int = len(intersection_vertices)
    int_verts_visited = [False for vert_idx in range(n_int)]
    intersections = []
    for int_vert_idx, is_visited in enumerate(int_verts_visited):
        # Skip if this vertex has been visited in another loop
        # ^ refer to exactly one visit rule.
        if is_visited:
            continue
        # To find a region start traversing along poly_1,
        # At intersections, pick a direction
        # that we know will lead us "inside" poly_2.
        start_idx = int_vert_idx + n_tot
        cur_idx = start_idx
        cur_poly = 1
        cur_dir = 0
        loop = []
        # If is_debug is set, print some useful information
        if is_debug:
            print("Loop idx:", start_idx)
        while True:
            # If is_debug is set, print some useful information
            if is_debug:
                print("{}-{}  ".format(\
                        "i" if cur_idx >= n_tot else "p" if cur_poly == 1 else "q",\
                        cur_idx), end='')
            # Add current vertex to the current loop
            if cur_idx >= n_tot:
                vert_pos = intersection_vertices[cur_idx-n_tot]
            elif cur_poly == 1:
                vert_pos = poly_1[cur_idx]
            else:
                vert_pos = poly_2[cur_idx]
            loop.append(vert_pos)
            # Is this vertex at an intersection?
            # If so, switch to the other polygon and start traversing
            # in the direction that goes into the current polygon.
            # And mark the current vertex as "visited"
            if cur_idx >= n_tot:
                int_verts_visited[cur_idx-n_tot] = True
                if cur_poly == 1:
                    cur_poly = 2
                    cur_dir = int(int_direction_db_2[cur_idx])
                else:
                    cur_poly = 1
                    cur_dir = int(int_direction_db_1[cur_idx])
            # Continue along the current polygon in the curent direction
            if cur_poly == 1:
                cur_idx = graph_1_edges[cur_idx][cur_dir]
            else:
                cur_idx = graph_2_edges[cur_idx][cur_dir]
            # If we wrapped around to the beginning exit the loop
            if cur_idx == start_idx:
                break
        if is_debug:
            print("\n")
        # Add the found loop to the intersection set
        intersections.append(loop)

    # Cast to integer coordinates unless otherwise stated
    dtype = np.float64 if float_coords else np.int32
    intersections = [np.array(intersection, dtype=dtype) for intersection in intersections]

    # Return all sorts of stuff if is_debug is set
    if is_debug:
        return n_tot, graph_1_edges, graph_2_edges, \
                intersection_vertices, \
                int_direction_db_1, int_direction_db_2, \
                intersections

    # If !is_debug, return intersections
    return intersections
