import pytrends.request as request
import time
import random
from Util import *
"""
Author: Cameron Knight
Copyright 2017, Cameron Knight, All rights reserved.
"""

trend_request = request.TrendReq(trends_username1, trends_password1, "PyTrends")
trend_request_low_level = request.TrendReq(trends_username2, trends_password2, "PyTrends2")


def process_name_and_get_data(name):
    global trend_request_low_level
    for i in range(len(name.split(' ')), 0, -1):
        try:
            if len(trend_request_low_level.suggestions(name)) > 0:
                try_name = trend_request_low_level.suggestions(name)[0]['title']
                num_relevant_terms = 0
                for w in name.split(' '):
                    if w in try_name:
                        num_relevant_terms += 1

                if num_relevant_terms > len(name.split(' ')) / 3:
                    name = try_name

            trend_request_low_level.build_payload([name])
            data = trend_request_low_level.interest_over_time()

        except:
            name = (''.join([x + ' ' for x in name.split(' ')[:i]])).strip()
        else:
            return name, data
    return None, None


def process_name(name):
    name, data = process_name_and_get_data(name)
    return name


def get_related_queries_and_data(name, depth=0):
    global trend_request

    related = [name]

    name = process_name(name)
    if name is None:
        return None, None

    failed = True
    attempts = 0
    while failed:
        try:
            failed = False
            attempts += 1
            trend_request.build_payload([name])
            related_queries = trend_request.related_queries()
            related = list(related_queries[name]['top']['query'])
            related += list(related_queries[name]['rising']['query'])
            processed_related = [name]
            for topic in related:
                processed_name, trend_data = process_name(topic)
                if processed_name is not None:
                    if depth > 0:
                        if DEBUG:
                            print("NEXT LEVEL: " + processed_name + ' Depth: ' + str(depth))
                        processed_related += get_related_queries_and_data(processed_name, depth-1)
                    else:
                        if DEBUG:
                            print("Term: " + processed_name)
                        processed_related += [[processed_name, trend_data]]
            return processed_related
        except:
            print('Failed', name)
            if attempts <= FETCH_ATTEMPTS:
                print("attempt num:", attempts)
                failed = True
            if attempts == 2:
                trend_request = request.TrendReq(trends_username1,
                                                 trends_password1,
                                                 "PyTrends" + str(random.randint(0, 10000)))
            time.sleep(FAIL_DELAY)

    return related


def get_related_queries(name, depth=0):
    """
    gets related querris, just names for a given word or phrase
    :param name: name or phrase to associate
    :param depth: depth of recursion between the term and its related querries
    :return: list of related terms
    """
    response = get_related_queries_and_data(name, depth)
    response = [i[0] for i in response]
    return response


def get_trend_data(name, depth=0):
    """
    gets a dictionary of terms and their corresponding data points
    :param name: the name to search
    :param depth: depth of recursion between terms and children
    :return: dictionary of terms and trend data for those terms
    """
    response = get_related_queries_and_data(name, depth)
    response = {i[0]: i[1] for i in response}
    return response

if __name__ == '__main__':
    print(get_trend_data(input("Enter a term"), 0))
